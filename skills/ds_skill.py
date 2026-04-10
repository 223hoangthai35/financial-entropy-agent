"""
Data Science / ML Layer -- Financial Entropy Agent
Phase Space Regime Classification via Full-Covariance GMM.
Plane 1 (Price): [WPE, SPE_Z] -> Raw features -> Full GMM -> Deterministic/Transitional/Stochastic.
Plane 2 (Volume): [Shannon, SampEn] -> PowerTransform -> Full GMM -> Consensus/Dispersed/Erratic.
Note: Plane 1 uses NO preprocessing — raw entropy topology is preserved by design.
"""

import logging
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer
from scipy.stats import normaltest

# NOTE: PowerTransformer va normaltest van duoc import cho VolumeRegimeClassifier.
# EntropyPhaseSpaceClassifier (Plane 1) KHONG dung PowerTransformer.

logger = logging.getLogger(__name__)


# ==============================================================================
# REGIME LABELS
# ==============================================================================
REGIME_NAMES: dict[int, str] = {
    0: "Deterministic",   # Lowest entropy — HIGH RISK (strong trend, crash/rally)
    1: "Transitional",   # Mid entropy    — MODERATE RISK (phase transition)
    2: "Stochastic",     # Highest entropy — LOW RISK (random walk, normal market)
}


# ==============================================================================
# ENTROPY PHASE SPACE CLASSIFIER (PLANE 1: [WPE, SPE_Z])
# ==============================================================================
class EntropyPhaseSpaceClassifier:
    """
    Plane 1 GMM Classifier trong Raw Entropy Phase Space.

    Pipeline:
        1. Feed RAW [WPE, SPE_Z] truc tiep vao GMM (KHONG transform).
        2. GaussianMixture(n=3, covariance_type='full')
           -> Moi cluster co ma tran hiep phuong sai RIENG (2x2).
              Full covariance tu dong xu ly scale khac nhau giua features.
        3. Semantic sorting: sum of centroid means (WPE_mean + SPE_Z_mean).
           Lowest combined entropy -> Stable, Mid -> Fragile, Highest -> Chaos.

    Tai sao KHONG dung PowerTransformer?
        - WPE bounded [0, 1], SPE_Z la Z-score (mean~0, std~1).
        - PowerTransform ep du lieu thanh Gaussian blob, pha huy
          topological boundaries tu nhien cua entropy metrics.
        - Full-covariance GMM du suc xu ly varying scales giua features.
        - Khac Plane 2 (Volume): Vol_Shannon/Vol_SampEn co phan phoi lech
          (right-skewed) nen Plane 2 can Yeo-Johnson truoc GMM.
          Plane 1 entropy metrics da co phan phoi phu hop GMM truc tiep.
    """

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",  # Moi cluster co covariance rieng
            n_init=10,
            max_iter=500,
            random_state=random_state,
        )
        self._cluster_to_regime: dict[int, int] = {}
        self.X_fitted: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "EntropyPhaseSpaceClassifier":
        """
        Fit Full GMM truc tiep tren raw [WPE, SPE_Z].
        """
        self.X_fitted = np.asarray(features, dtype=np.float64)
        self.gmm.fit(self.X_fitted)
        self._map_clusters_by_combined_entropy()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict semantic regime labels (0=Stable, 1=Fragile, 2=Chaos)."""
        raw_labels = self.gmm.predict(features)
        mapped = np.array([self._cluster_to_regime.get(l, l) for l in raw_labels])
        return mapped

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit + predict trong 1 buoc."""
        self.fit(features)
        raw_labels = self.gmm.predict(self.X_fitted)
        mapped = np.array([self._cluster_to_regime.get(l, l) for l in raw_labels])
        return mapped

    def get_regime_name(self, label: int) -> str:
        """Chuyen semantic label thanh ten regime."""
        return REGIME_NAMES.get(label, f"Unknown_{label}")

    def get_ellipse_params(self, cluster_idx: int, n_std: float = 2.0) -> dict:
        """
        Tinh tham so ellipse (95% confidence) cho cluster trong raw space.
        Voi covariance_type='full', gmm.covariances_ la (n_components, n_features, n_features).
        Moi cluster co ellipse RIENG voi hinh dang khac nhau.
        Returns: {"center": (cx, cy), "width": w, "height": h, "angle": theta_deg}
        """
        mean = self.gmm.means_[cluster_idx]
        cov = self.gmm.covariances_[cluster_idx]

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])
        return {
            "center": (float(mean[0]), float(mean[1])),
            "width": float(width),
            "height": float(height),
            "angle": float(angle),
        }

    def _map_clusters_by_combined_entropy(self) -> None:
        """
        Map GMM cluster indices -> semantic regime labels (0,1,2)
        dua tren tong cac centroid means (WPE_mean + SPE_Z_mean).
        Lowest combined entropy -> 0 (Stable).
        Highest combined entropy -> 2 (Chaos).
        """
        combined = self.gmm.means_.sum(axis=1)  # WPE + SPE_Z per cluster
        sorted_indices = np.argsort(combined)
        self._cluster_to_regime = {
            int(sorted_indices[i]): i for i in range(self.n_components)
        }


# ==============================================================================
# CONVENIENCE: FIT + PREDICT (FUNCTIONAL API) -- PRICE PLANE
# ==============================================================================
def fit_predict_regime(
    features: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, EntropyPhaseSpaceClassifier]:
    """
    Ham tien ich: tao EntropyPhaseSpaceClassifier, fit va predict.
    Input:  features (N x 2 array: [WPE, SPE_Z])
    Output: (labels, fitted_classifier)
    Labels duoc quyet dinh boi Full GMM trong Entropy Phase Space.
    """
    clf = EntropyPhaseSpaceClassifier(n_components=n_components)
    labels = clf.fit_predict(features)
    return labels, clf


# ==============================================================================
# VOLUME REGIME LABELS (PLANE 2: LIQUIDITY STRUCTURE)
# ==============================================================================
VOLUME_REGIME_NAMES: dict[int, str] = {
    0: "Consensus Flow",
    1: "Dispersed Flow",
    2: "Erratic/Noisy Flow",
}


# ==============================================================================
# VOLUME REGIME CLASSIFIER (PLANE 2)
# ==============================================================================
class VolumeRegimeClassifier:
    """
    GMM Unsupervised cho Volume Entropy Plane.
    Features dau vao: [Vol_Shannon, Vol_SampEn].
    Mapping: sap xep cluster theo tong centroid (Shannon + SampEn).
    """

    def __init__(self, n_components: int = 3, random_state: int = 42) -> None:
        self.n_components = n_components
        self.power_tf = PowerTransformer(method="yeo-johnson", standardize=True)
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=10,
            random_state=random_state,
        )
        self._cluster_to_regime: dict[int, str] = {}

    def fit(self, features: np.ndarray) -> "VolumeRegimeClassifier":
        """Fit PowerTransformer + GMM tren [Vol_Shannon, Vol_SampEn] (N x 2)."""
        self.X_transformed = self.power_tf.fit_transform(features)
        self.gmm.fit(self.X_transformed)
        self._map_clusters(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict volume regime labels."""
        X_tf = self.power_tf.transform(features)
        return self.gmm.predict(X_tf)

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit + predict trong 1 buoc."""
        self.fit(features)
        return self.gmm.predict(self.X_transformed)

    def get_regime_name(self, label: int) -> str:
        """Chuyen cluster index thanh ten volume regime."""
        return self._cluster_to_regime.get(label, f"Unknown_{label}")

    def _map_clusters(self, original_features: np.ndarray) -> None:
        """Mapping: centroid sort tren original space (sum of Shannon + SampEn)."""
        labels = self.gmm.predict(self.X_transformed)
        centroids = np.array([
            original_features[labels == k].mean(axis=0)
            for k in range(self.n_components)
        ])
        sort_key = centroids.sum(axis=1)
        sorted_indices = np.argsort(sort_key)

        regime_order = list(VOLUME_REGIME_NAMES.values())
        self._cluster_to_regime = {
            int(sorted_indices[i]): regime_order[min(i, len(regime_order) - 1)]
            for i in range(self.n_components)
        }


# ==============================================================================
# CONVENIENCE: FIT + PREDICT (FUNCTIONAL API) -- VOLUME PLANE
# ==============================================================================
def fit_predict_volume_regime(
    features: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, VolumeRegimeClassifier]:
    """
    Ham tien ich: tao VolumeRegimeClassifier, fit va predict.
    Input:  features (N x 2 array: [Vol_Shannon, Vol_SampEn])
    Output: (labels, fitted_classifier)
    """
    clf = VolumeRegimeClassifier(n_components=n_components)
    labels = clf.fit_predict(features)
    return labels, clf


# ==============================================================================
# TESTING BLOCK
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
    np.random.seed(42)

    print("=" * 60)
    print("TEST 1: Plane 1 -- Raw Entropy Phase Space (Full GMM, NO transform)")
    print("=" * 60)

    # Tao 3 cum gia lap: WPE (X, [0,1]) va SPE_Z (Y, z-score)
    stable = np.column_stack([
        np.random.beta(5, 8, 300) * 0.3 + 0.3,   # WPE ~0.3-0.6
        np.random.randn(300) * 0.4 - 0.8          # SPE_Z low (near -1)
    ])
    fragile = np.column_stack([
        np.random.beta(5, 5, 200) * 0.2 + 0.6,    # WPE ~0.6-0.8
        np.random.randn(200) * 0.5                 # SPE_Z mid (near 0)
    ])
    chaos = np.column_stack([
        np.random.beta(8, 3, 100) * 0.15 + 0.82,  # WPE ~0.82-0.97
        np.random.randn(100) * 0.6 + 0.8           # SPE_Z high (near +1)
    ])

    fake_features = np.vstack([stable, fragile, chaos])
    print(f"  Feature matrix shape : {fake_features.shape}")
    print(f"  Columns semantics    : [WPE (raw, [0,1]), SPE_Z (raw, z-score)]")

    labels, clf = fit_predict_regime(fake_features, n_components=3)

    print(f"\n  GMM covariance_type  : {clf.gmm.covariance_type}")
    print(f"  PowerTransformer     : NONE (raw features)")

    print(f"\n  GMM Centroids (Raw Phase Space):")
    for i in range(3):
        regime_idx = clf._cluster_to_regime.get(i, i)
        name = clf.get_regime_name(regime_idx)
        mean = clf.gmm.means_[i]
        combined = mean.sum()
        print(f"    Cluster {i} -> {name:15s} : WPE={mean[0]:.3f}, SPE_Z={mean[1]:+.3f}, sum={combined:+.3f}")

    print(f"\n  Combined-entropy sorting check:")
    combined_scores = clf.gmm.means_.sum(axis=1)
    print(f"    Combined scores: {[f'{s:+.3f}' for s in combined_scores]}")

    print(f"\n  Predicted labels     : {np.unique(labels)}")
    print(f"  Label distribution   :")
    for lbl in np.unique(labels):
        name = clf.get_regime_name(lbl)
        count = (labels == lbl).sum()
        print(f"    {lbl} -> {name:25s} (n={count})")

    # Ellipse params (full covariance in RAW space)
    print(f"\n  95% Confidence Ellipses (FULL, RAW space):")
    for i in range(3):
        e = clf.get_ellipse_params(i, n_std=2.0)
        regime_idx = clf._cluster_to_regime.get(i, i)
        name = clf.get_regime_name(regime_idx)
        print(f"    {name:15s}: center=({e['center'][0]:.3f},{e['center'][1]:+.3f}), w={e['width']:.3f}, h={e['height']:.3f}, angle={e['angle']:.1f}")

    print()
    print("=" * 60)
    print("TEST 2: Plane 2 -- Volume Regime (PowerTransform + GMM)")
    print("=" * 60)

    consensus = np.random.randn(50, 2) * 0.05 + np.array([0.65, 0.8])
    dispersed = np.random.randn(50, 2) * 0.05 + np.array([0.85, 1.2])
    erratic = np.random.randn(50, 2) * 0.05 + np.array([0.95, 2.5])

    vol_features = np.vstack([consensus, dispersed, erratic])
    print(f"  Feature matrix shape : {vol_features.shape}")

    vol_labels, vol_clf = fit_predict_volume_regime(vol_features, n_components=3)

    print(f"\n  Predicted labels     : {np.unique(vol_labels)}")
    for lbl in np.unique(vol_labels):
        name = vol_clf.get_regime_name(lbl)
        count = (vol_labels == lbl).sum()
        print(f"    {lbl} -> {name:25s} (n={count})")
