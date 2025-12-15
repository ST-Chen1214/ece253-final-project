# === file: segmentation_bayes.py ===
import numpy as np
import cv2
from scipy.stats import multivariate_normal

#############################
# 1. Extract 8x8 DCT features
#############################

def extract_dct_patches(img_gray, patch_size=8, stride=8):
    """
    img_gray: float32 [0,1] or uint8.
    Returns:
      features: (N_patches, patch_size*patch_size) DCT coefficients
      coords:   (N_patches, 2) top-left coordinates (y,x) of each patch
    """
    if img_gray.dtype != np.float32:
        img_gray = img_gray.astype(np.float32) / 255.0

    H, W = img_gray.shape
    feats = []
    coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img_gray[y:y + patch_size, x:x + patch_size]
            dct = cv2.dct(patch)
            feats.append(dct.flatten())
            coords.append((y, x))
    return np.array(feats, np.float32), np.array(coords, np.int32)


#############################
# 2. Bayesian Gaussian classifier
#############################

class BayesianGaussianClassifier:
    """
    Simple Gaussian classifier:
      P(x | class) ~ N(mu_c, Sigma_c)
    Prior estimated from class frequency.
    """

    def __init__(self, reg_eps=1e-4):
        self.reg_eps = reg_eps
        self.class_means = {}
        self.class_covs = {}
        self.class_priors = {}
        self.classes_ = []

    def fit(self, X, y):
        """
        X: (N, D)
        y: (N,) integer labels {0,...,C-1}
        """
        self.classes_ = np.unique(y)
        N = len(X)
        for c in self.classes_:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            cov = np.cov(Xc.T)
            cov += self.reg_eps * np.eye(cov.shape[0])
            self.class_means[c] = mu
            self.class_covs[c] = cov
            self.class_priors[c] = float(len(Xc)) / N

    def predict_proba(self, X):
        """
        Returns P(class|X) for each sample.
        """
        N = len(X)
        C = len(self.classes_)
        P = np.zeros((N, C), dtype=np.float64)

        for idx_c, c in enumerate(self.classes_):
            rv = multivariate_normal(
                mean=self.class_means[c],
                cov=self.class_covs[c]
            )
            px_given_c = rv.pdf(X)
            P[:, idx_c] = px_given_c * self.class_priors[c]

        P_sum = P.sum(axis=1, keepdims=True) + 1e-12
        P /= P_sum
        return P

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


#############################
# 3. Build training data from masks
#############################

def build_training_data_from_masks(img_paths, mask_paths,
                                   patch_size=8, stride=8):
    """
    img_paths: list of RGB/BGR images
    mask_paths: corresponding single-channel masks (0=background, 1=food, or multi-class)
    Returns: X (features), y (labels)
    """
    all_feats = []
    all_labels = []
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Cannot read mask: {mask_path}")
            continue

        H, W = gray.shape
        feats, coords = extract_dct_patches(gray,
                                            patch_size=patch_size,
                                            stride=stride)
        labels = []
        for (y, x) in coords:
            patch_mask = mask[y:y + patch_size, x:x + patch_size]
            # majority vote
            vals, counts = np.unique(patch_mask, return_counts=True)
            labels.append(int(vals[np.argmax(counts)]))

        all_feats.append(feats)
        all_labels.append(np.array(labels, dtype=np.int32))

    X = np.vstack(all_feats)
    y = np.concatenate(all_labels)
    return X, y


#############################
# 4. Apply segmentation to new image
#############################

def segment_image_dct_bayes(img_bgr, classifier,
                            patch_size=8, stride=8):
    """
    img_bgr: input BGR image
    classifier: trained BayesianGaussianClassifier
    Returns:
      label_map: (H,W) int32, class index per pixel
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    feats, coords = extract_dct_patches(img_gray,
                                        patch_size=patch_size,
                                        stride=stride)
    labels = classifier.predict(feats)

    H, W = img_gray.shape
    label_map = np.zeros((H, W), np.int32)
    count_map = np.zeros((H, W), np.int32)

    for (label, (y, x)) in zip(labels, coords):
        label_map[y:y + patch_size, x:x + patch_size] += label
        count_map[y:y + patch_size, x:x + patch_size] += 1

    count_map[count_map == 0] = 1
    label_map = (label_map / count_map).round().astype(np.int32)
    return label_map
