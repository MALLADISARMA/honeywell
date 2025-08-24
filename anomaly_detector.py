import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MultivariateTimeSeriesAnomalyDetector:
    """
    Multivariate Time Series Anomaly Detection System.
    
    This class detects anomalies in multivariate time series data 
    using an ensemble of methods:
        - Isolation Forest
        - Statistical models (Elliptic Envelope / Z-score fallback)
        - PCA reconstruction error
        - Clustering (DBSCAN)

    It also attributes anomalies to top contributing original features 
    (so we know which variables influenced the anomaly).
    """

    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize anomaly detector.

        Parameters:
        contamination (float): Expected proportion of anomalies in the dataset.
        random_state (int): Random seed for reproducibility.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()    # Standardization for stable training
        self.feature_importance = {}      # Stores correlation of features with anomaly scores
        self.is_fitted = False
        self.original_features = []       # List of original dataset features used

    # ---------------- Feature Engineering ---------------- #
    def _create_time_features(self, df, timestamp_col=None):
        """
        Add time-based features (hour, day, month, weekend) from a timestamp column.
        """
        df_enhanced = df.copy()
        if timestamp_col and timestamp_col in df.columns:
            try:
                df_enhanced[timestamp_col] = pd.to_datetime(df_enhanced[timestamp_col], errors='coerce')
                df_enhanced['hour'] = df_enhanced[timestamp_col].dt.hour
                df_enhanced['day_of_week'] = df_enhanced[timestamp_col].dt.dayofweek
                df_enhanced['month'] = df_enhanced[timestamp_col].dt.month
                df_enhanced['is_weekend'] = df_enhanced['day_of_week'].isin([5, 6]).astype(int)
            except Exception as e:
                print(f"Warning: Could not parse timestamp column: {e}")
        return df_enhanced

    def _create_statistical_features(self, df, window_sizes=[3, 5, 10]):
        """
        Add rolling mean, std, and deviation features for numeric columns.
        This captures short-term trends and fluctuations.
        """
        df_stats = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            for window in window_sizes:
                if window < len(df):
                    # Rolling mean and std
                    df_stats[f'{col}_roll_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    df_stats[f'{col}_roll_std_{window}'] = df[col].rolling(window, min_periods=1).std()

                    # Standardized deviation from rolling mean
                    rolling_mean = df_stats[f'{col}_roll_mean_{window}']
                    rolling_std = df_stats[f'{col}_roll_std_{window}']
                    df_stats[f'{col}_deviation_{window}'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

        # Fill missing values from rolling computations
        return df_stats.fillna(method='bfill').fillna(method='ffill').fillna(0)

    # ---------------- Anomaly Detection Methods ---------------- #
    def _detect_isolation_forest(self, X):
        """
        Detect anomalies using Isolation Forest (tree-based outlier detection).
        Returns normalized anomaly scores (0–100).
        """
        iso = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        scores = -iso.fit(X).decision_function(X)
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _detect_statistical(self, X):
        """
        Detect anomalies using Elliptic Envelope (robust covariance estimation).
        Fallback: Z-score method if fitting fails.
        """
        try:
            env = EllipticEnvelope(contamination=self.contamination, random_state=self.random_state)
            scores = -env.fit(X).decision_function(X)
        except:
            # fallback: z-score mean
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
            scores = np.mean(z_scores, axis=1)

        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _detect_pca(self, X):
        """
        Detect anomalies using PCA reconstruction error.
        Measures how well PCA can reconstruct the data.
        """
        n_samples, n_features = X.shape
        n_components = min(n_features, max(1, int(n_samples * 0.8)), 10)

        try:
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_rec = pca.inverse_transform(pca.fit_transform(X))
            errors = np.sum((X - X_rec) ** 2, axis=1)  # reconstruction error
            scores = errors
        except:
            scores = np.var(X, axis=1)  # fallback variance
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _detect_clustering(self, X):
        """
        Detect anomalies using DBSCAN clustering.
        Outliers are labeled as -1 by DBSCAN.
        """
        try:
            # Approximate eps using distances between nearby points
            eps = np.percentile(
                [np.linalg.norm(X[i] - X[j]) for i in range(len(X)) for j in range(i+1, min(i+10, len(X)))],
                75
            )
            db = DBSCAN(eps=eps, min_samples=max(2, int(len(X) * 0.05)))
            labels = db.fit_predict(X)

            # Anomalies (label=-1) get high scores, others based on distance to cluster center
            scores = np.array([
                100 if l == -1 else np.linalg.norm(x - X[labels == l].mean(axis=0)) * 50
                for x, l in zip(X, labels)
            ])
        except:
            # fallback: distance from global mean
            center = X.mean(axis=0)
            scores = np.linalg.norm(X - center, axis=1)

        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    # ---------------- Feature Attribution ---------------- #
    def _get_top_features(self, row_index, X_original, features, top_n=3):
        """
        Identify top contributing features for a given row
        based on z-score deviation from mean.
        """
        if row_index >= len(X_original):
            return []

        row = X_original[row_index]
        mean = X_original.mean(axis=0)
        std = X_original.std(axis=0) + 1e-8
        z_scores = np.abs((row - mean) / std)

        idxs = np.argsort(z_scores)[-top_n:][::-1]
        return [f"{features[i]}({z_scores[i]:.2f})" for i in idxs if i < len(features)]

    # ---------------- Main Pipeline ---------------- #
    def fit_predict(self, df, timestamp_col=None, exclude_cols=None, normal_start=None, normal_end=None):
        """
        Full pipeline: feature engineering, anomaly detection, feature attribution.

        Parameters:
        df (pd.DataFrame): Input dataset
        timestamp_col (str): Column name for timestamp (optional)
        exclude_cols (list): Columns to exclude from features
        normal_start, normal_end: Range of normal period for training

        Returns:
        pd.DataFrame: Original data + anomaly_score + is_anomaly + top_contributing_features
        """
        print(f"Processing dataset with shape: {df.shape}")

        # Handle excluded columns
        if exclude_cols is None:
            exclude_cols = []
        if timestamp_col and timestamp_col not in exclude_cols:
            exclude_cols.append(timestamp_col)

        # Create engineered features
        df_feat = self._create_statistical_features(self._create_time_features(df, timestamp_col))
        feature_cols = [c for c in df_feat.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        if not feature_cols:
            raise ValueError("No numeric features found for analysis")

        # Preserve original numeric features for attribution
        numeric_cols_original = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
        X_original = np.nan_to_num(df[numeric_cols_original].values, nan=0.0, posinf=0.0, neginf=0.0)
        self.original_features = numeric_cols_original

        # Processed features for anomaly detection
        X = np.nan_to_num(df_feat[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)

        # Define training period = normal data
        if normal_start and normal_end and timestamp_col:
            mask = (df[timestamp_col] >= normal_start) & (df[timestamp_col] <= normal_end)
            X_train = X[mask]
        elif normal_start is not None and normal_end is not None:
            X_train = X[normal_start:normal_end]
        else:
            X_train = X[:int(0.7 * len(df))]  # default: 70% first rows as normal

        # Scale features
        try:
            X_train_s = self.scaler.fit_transform(X_train)
            X_s = self.scaler.transform(X)
        except:
            X_s = X

        self.is_fitted = True

        # Ensemble anomaly score (weighted average of methods)
        scores = (
            0.3 * self._detect_isolation_forest(X_s) +
            0.25 * self._detect_statistical(X_s) +
            0.25 * self._detect_pca(X_s) +
            0.2 * self._detect_clustering(X_s)
        )

        # Feature importance: correlation between features and anomaly score
        self.feature_importance = {
            f: np.corrcoef(X_original[:, i], scores)[0, 1] 
            for i, f in enumerate(numeric_cols_original)
        }

        # Build results DataFrame
        results = df.copy()
        results['anomaly_score'] = np.round(scores, 2)

        # Define anomaly threshold at 90th percentile
        threshold = np.percentile(scores, 90)
        results['is_anomaly'] = (scores >= threshold).astype(int)

        # Compute top contributing features per row
        top_feats = []
        for i in range(len(df)):
            tf = self._get_top_features(i, X_original, numeric_cols_original)
            top_feats.append(", ".join(tf) if tf else "N/A")
        results['top_contributing_features'] = top_feats

        print(f"Anomaly detection completed. Found {results['is_anomaly'].sum()} anomalies.")
        return results
