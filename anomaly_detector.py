import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.covariance import EllipticEnvelope
from scipy import stats
import warnings

# Suppress all warnings for cleaner output in production environments.
warnings.filterwarnings("ignore")


class MultivariateTimeSeriesAnomalyDetector:
    """
    A robust anomaly detection class for multivariate time series data.

    This class leverages an ensemble of different machine learning models
    (Isolation Forest, Elliptic Envelope, PCA, and Clustering) to provide
    a comprehensive anomaly score. It automatically performs feature
    engineering by creating time-based and statistical features.

    Parameters:
    - contamination (float): The proportion of outliers in the data set.
                             Used by IsolationForest and EllipticEnvelope.
                             Defaults to 0.1.
    - random_state (int): Seed for reproducibility. Defaults to 42.
    - max_rows (int): Maximum number of rows to sample from the dataset.
                      This limits processing time for very large files.
                      Defaults to 10000.
    """

    def __init__(self, contamination=0.1, random_state=42, max_rows=10000):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.original_features = []
        self.max_rows = max_rows

    def _create_time_features(self, df, timestamp_col=None):
        """
        Generates time-based features from a timestamp column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            timestamp_col (str, optional): The name of the timestamp column.

        Returns:
            pd.DataFrame: The DataFrame with new time-based features.
        """
        df_enhanced = df.copy()
        if timestamp_col and timestamp_col in df.columns:
            try:
                # Convert the column to datetime objects
                df_enhanced[timestamp_col] = pd.to_datetime(
                    df_enhanced[timestamp_col], errors="coerce"
                )
                # Extract temporal features
                df_enhanced["hour"] = df_enhanced[timestamp_col].dt.hour
                df_enhanced["day_of_week"] = df_enhanced[timestamp_col].dt.dayofweek
                df_enhanced["month"] = df_enhanced[timestamp_col].dt.month
                df_enhanced["is_weekend"] = (
                    df_enhanced["day_of_week"].isin([5, 6]).astype(int)
                )
            except Exception as e:
                print(f"Warning: Could not parse timestamp column '{timestamp_col}': {e}")
        return df_enhanced

    def _create_statistical_features(self, df, window_sizes=None):
        """
        Generates rolling statistical features (mean and standard deviation).

        Args:
            df (pd.DataFrame): The input DataFrame.
            window_sizes (list, optional): A list of window sizes for rolling calculations.
                                           Defaults to [3, 5].

        Returns:
            pd.DataFrame: The DataFrame with new statistical features.
        """
        if window_sizes is None:
            window_sizes = [3, 5]
        df_stats = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            for window in window_sizes:
                if window < len(df):
                    roll_mean = df[col].rolling(window, min_periods=1).mean()
                    roll_std = df[col].rolling(window, min_periods=1).std()
                    df_stats[f"{col}_roll_mean_{window}"] = roll_mean
                    df_stats[f"{col}_roll_std_{window}"] = roll_std
        return df_stats.fillna(0)

    def _detect_isolation_forest(self, X):
        """Detects anomalies using Isolation Forest."""
        iso = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=50,
            max_samples=min(5000, X.shape[0]),
            n_jobs=-1
        )
        scores = -iso.fit(X).decision_function(X)
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _detect_statistical(self, X):
        """
        Detects anomalies using Elliptic Envelope with a Z-score fallback.

        This method identifies points that lie outside a defined statistical
        distribution. It's particularly useful for data with a clear
        Gaussian-like distribution.
        """
        try:
            env = EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state
            )
            scores = -env.fit(X).decision_function(X)
        except Exception:
            # Fallback to a Z-score-based approach for robustness
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy="omit"))
            scores = np.mean(z_scores, axis=1)
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _detect_pca(self, X):
        """
        Detects anomalies using Principal Component Analysis (PCA).

        Anomalies are identified as data points that have high reconstruction
        error when projected onto a lower-dimensional space and back.
        """
        try:
            pca = PCA(
                n_components=min(5, X.shape[1]),
                svd_solver="randomized",
                random_state=self.random_state
            )
            X_rec = pca.inverse_transform(pca.fit_transform(X))
            errors = np.sum((X - X_rec) ** 2, axis=1)
            scores = errors
        except Exception:
            # Fallback to a simple variance-based score
            scores = np.var(X, axis=1)
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _detect_clustering(self, X):
        """
        Detects anomalies using clustering-based methods (MiniBatchKMeans or DBSCAN).

        Anomalies are identified as points far from cluster centers (MiniBatchKMeans)
        or as noise points (DBSCAN).
        """
        try:
            if X.shape[0] > 5000:
                kmeans = MiniBatchKMeans(n_clusters=8, batch_size=1000,
                                         random_state=self.random_state)
                labels = kmeans.fit_predict(X)
                centers = kmeans.cluster_centers_
                scores = np.linalg.norm(X - centers[labels], axis=1)
            else:
                sample = X[np.random.choice(len(X), min(200, len(X)), replace=False)]
                dists = [np.linalg.norm(sample[i] - sample[j])
                         for i in range(len(sample)) for j in range(i + 1, len(sample))]
                eps = np.percentile(dists, 75)
                db = DBSCAN(eps=eps, min_samples=5, n_jobs=-1)
                labels = db.fit_predict(X)
                scores = np.array([
                    100 if l == -1
                    else np.linalg.norm(x - X[labels == l].mean(axis=0))
                    for x, l in zip(X, labels)
                ])
        except Exception:
            # Fallback to a simple distance-from-mean score
            center = X.mean(axis=0)
            scores = np.linalg.norm(X - center, axis=1)
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _get_top_features(self, row_index, X_original, features):
        """Identifies the feature that contributes most to a given anomaly."""
        row = X_original[row_index]
        mean = X_original.mean(axis=0)
        std = X_original.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
        z_scores = np.abs((row - mean) / std)
        idx = np.argmax(z_scores)
        return f"{features[idx]}(Z-score: {z_scores[idx]:.2f})"

    def fit_predict(self, df, timestamp_col=None, exclude_cols=None):
        """
        Runs the anomaly detection process on the input DataFrame.

        Args:
            df (pd.DataFrame): The input time series DataFrame.
            timestamp_col (str, optional): The name of the timestamp column.
                                           Defaults to "timestamp".
            exclude_cols (list, optional): A list of columns to exclude from
                                           the analysis.

        Returns:
            pd.DataFrame: The original DataFrame with added anomaly scores
                          and a binary 'is_anomaly' column.
        """
        # Data sampling to control execution time
        if len(df) > self.max_rows:
            df = df.sample(self.max_rows, random_state=self.random_state).reset_index(drop=True)

        if exclude_cols is None:
            exclude_cols = []
        if timestamp_col and timestamp_col not in exclude_cols:
            exclude_cols.append(timestamp_col)

        # Feature engineering
        df_feat = self._create_statistical_features(
            self._create_time_features(df, timestamp_col)
        )
        feature_cols = [c for c in df_feat.select_dtypes(include=[np.number]).columns
                        if c not in exclude_cols]

        numeric_cols_original = [c for c in df.select_dtypes(include=[np.number]).columns
                                 if c not in exclude_cols]
        X_original = np.nan_to_num(df[numeric_cols_original].values)
        self.original_features = numeric_cols_original

        X = np.nan_to_num(df_feat[feature_cols].values)
        X_train = X[: int(0.7 * len(df))]

        # Data scaling
        X_train_s = self.scaler.fit_transform(X_train)
        X_s = self.scaler.transform(X)

        # Individual model scoring
        iso_score = self._detect_isolation_forest(X_s)
        stat_score = self._detect_statistical(X_s)
        pca_score = self._detect_pca(X_s)
        cluster_score = self._detect_clustering(X_s)

        # Final weighted score
        final_score = (
            0.3 * iso_score +
            0.25 * stat_score +
            0.25 * pca_score +
            0.2 * cluster_score
        )

        # Anomaly thresholding and results generation
        threshold = np.percentile(final_score, 90)
        results = df.copy()
        results["iso_score"] = np.round(iso_score, 2)
        results["stat_score"] = np.round(stat_score, 2)
        results["pca_score"] = np.round(pca_score, 2)
        results["cluster_score"] = np.round(cluster_score, 2)
        results["final_score"] = np.round(final_score, 2)
        results["threshold"] = threshold
        results["is_anomaly"] = (final_score >= threshold).astype(int)
        results["top_feature"] = [
            self._get_top_features(i, X_original, numeric_cols_original)
            for i in range(len(df))
        ]

        print(f"Anomaly detection completed. Found {results['is_anomaly'].sum()} anomalies.")
        return results


# ---------------- Main Execution Block ---------------- #
if __name__ == "__main__":
    try:
        file_path = input("Enter file path (CSV or Excel): ").strip()
        
        # Determine file type and read data
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            data = pd.read_excel(file_path, sheet_name="Sheet4")
        else:
            raise ValueError("File must be a CSV (.csv) or Excel (.xlsx) file.")

        # Initialize and run the detector
        detector = MultivariateTimeSeriesAnomalyDetector(contamination=0.1)
        result = detector.fit_predict(data, timestamp_col="timestamp")

        # Save the results
        result.to_csv("anomaly_results.csv", index=False)
        print("\nResults saved to anomaly_results.csv")

    except Exception as e:
        print(f"An error occurred: {e}")
