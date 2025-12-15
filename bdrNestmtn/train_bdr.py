# === file: train_bdr.py ===
import os
import glob
import cv2
import joblib

from segmentation_bayes import (
    BayesianGaussianClassifier,
    build_training_data_from_masks,
)

def find_mask_for_image(img_path, mask_dir):
    """
    Given an image path, find the corresponding mask file in mask_dir
    by matching the filename stem (without extension).
    """
    img_name = os.path.basename(img_path)
    stem = os.path.splitext(img_name)[0]

    patterns = [
        os.path.join(mask_dir, stem + ".png"),
        os.path.join(mask_dir, stem + ".jpg"),
        os.path.join(mask_dir, stem + ".jpeg"),
        os.path.join(mask_dir, stem + ".bmp"),
    ]

    for p in patterns:
        if os.path.exists(p):
            return p
    return None


def main():
    train_img_dir = os.path.join("datasets", "train_images")
    train_mask_dir = os.path.join("datasets", "train_masks")

    # Collect all training images
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_paths.extend(glob.glob(os.path.join(train_img_dir, ext)))
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        raise RuntimeError(f"No training images found in {train_img_dir}")

    mask_paths = []
    for img_path in img_paths:
        mask_path = find_mask_for_image(img_path, train_mask_dir)
        if mask_path is None:
            raise RuntimeError(
                f"No mask found for image {img_path}. "
                f"Please make sure filenames match between train_images and train_masks."
            )
        mask_paths.append(mask_path)

    print(f"[INFO] Found {len(img_paths)} image/mask pairs for training.")

    # Build training data
    X, y = build_training_data_from_masks(img_paths, mask_paths,
                                          patch_size=8, stride=8)
    print(f"[INFO] Training data shape: X={X.shape}, y={y.shape}")
    print(f"[INFO] Unique labels in y: {sorted(set(y))}")

    # Optional: if masks are grayscale 0~255, you may want to binarize here
    # For example, treat >0 as food (1), 0 as background (0)
    # y = (y > 0).astype(int)

    # Train classifier
    clf = BayesianGaussianClassifier(reg_eps=1e-4)
    clf.fit(X, y)
    print("[INFO] Finished training Bayesian Gaussian classifier.")

    # Save model
    model_path = "bdr_model.pkl"
    joblib.dump(clf, model_path)
    print(f"[INFO] Saved model to {model_path}")


if __name__ == "__main__":
    main()
