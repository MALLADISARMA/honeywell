import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.covariance import EllipticEnvelope
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class MultivariateTimeSeriesAnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42, max_rows=10000):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.original_features = []
        self.max_rows = max_rows

    def _create_time_features(self, df, timestamp_col=None):
        df_enhanced = df.copy()
        if timestamp_col and timestamp_col in df.columns:
            try:
                df_enhanced[timestamp_col] = pd.to_datetime(
                    df_enhanced[timestamp_col], errors="coerce"
                )
                df_enhanced["hour"] = df_enhanced[timestamp_col].dt.hour
                df_enhanced["day_of_week"] = df_enhanced[timestamp_col].dt.dayofweek
                df_enhanced["month"] = df_enhanced[timestamp_col].dt.month
                df_enhanced["is_weekend"] = (
                    df_enhanced["day_of_week"].isin([5, 6]).astype(int)
                )
            except Exception as e:
                print(f"Warning: Could not parse timestamp column: {e}")
        return df_enhanced

    def _create_statistical_features(self, df, window_sizes=None):
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
        try:
            env = EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state
            )
            scores = -env.fit(X).decision_function(X)
        except Exception:
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy="omit"))
            scores = np.mean(z_scores, axis=1)
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _detect_pca(self, X):
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
            scores = np.var(X, axis=1)
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _detect_clustering(self, X):
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
            center = X.mean(axis=0)
            scores = np.linalg.norm(X - center, axis=1)
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten() * 100

    def _get_top_features_list(self, row_index, X_original, features, top_n=7):
        """
        Returns a list of top N features with highest z-scores for a given row.
        If fewer than top_n features, fills remaining with empty strings.
        """
        row = X_original[row_index]
        mean = X_original.mean(axis=0)
        std = X_original.std(axis=0) + 1e-8
        z_scores = np.abs((row - mean) / std)
        top_indices = np.argsort(-z_scores)[:top_n]
        top_features = [f"{features[i]}({z_scores[i]:.2f})" for i in top_indices]
        while len(top_features) < top_n:
            top_features.append("")
        return top_features

    def fit_predict(self, df, timestamp_col=None, exclude_cols=None):
        if len(df) > self.max_rows:
            df = df.sample(self.max_rows, random_state=self.random_state).reset_index(drop=True)

        if exclude_cols is None:
            exclude_cols = []
        if timestamp_col and timestamp_col not in exclude_cols:
            exclude_cols.append(timestamp_col)

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

        X_train_s = self.scaler.fit_transform(X_train)
        X_s = self.scaler.transform(X)

        # Individual scores
        iso_score = self._detect_isolation_forest(X_s)
        stat_score = self._detect_statistical(X_s)
        pca_score = self._detect_pca(X_s)
        cluster_score = self._detect_clustering(X_s)

        # Final score
        final_score = (
            0.3 * iso_score +
            0.25 * stat_score +
            0.25 * pca_score +
            0.2 * cluster_score
        )

        threshold = np.percentile(final_score, 90)
        results = df.copy()
        results["iso_score"] = np.round(iso_score, 2)
        results["stat_score"] = np.round(stat_score, 2)
        results["pca_score"] = np.round(pca_score, 2)
        results["cluster_score"] = np.round(cluster_score, 2)
        results["final_score"] = np.round(final_score, 2)
        results["threshold"] = threshold
        results["is_anomaly"] = (final_score >= threshold).astype(int)

        # Top 7 features
        top_n = 7
        top_features_list = [
            self._get_top_features_list(i, X_original, numeric_cols_original, top_n)
            for i in range(len(df))
        ]
        for i in range(top_n):
            results[f"top_feature_{i+1}"] = [row[i] for row in top_features_list]

        print(f"Anomaly detection completed. Found {results['is_anomaly'].sum()} anomalies.")
        return results


# ---------------- Main Run ---------------- #
if __name__ == "__main__":
    try:
        file_path = input("Enter file path (CSV or Excel): ").strip()
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            data = pd.read_excel(file_path, sheet_name="Sheet4")
        else:
            raise ValueError("File must be CSV or Excel (.xlsx)")

        detector = MultivariateTimeSeriesAnomalyDetector(contamination=0.1)
        result = detector.fit_predict(data, timestamp_col="timestamp")

        result.to_csv("anomaly_results.csv", index=False)
        print("\nResults saved to anomaly_results.csv")

    except Exception as e:
        print(f"Error: {e}")
