# === file: run_bdr_and_calories.py ===
import os
import glob
import csv

import cv2
import numpy as np
import joblib

from segmentation_bayes import segment_image_dct_bayes
from calorie_estimation import estimate_calories

# ---- User-configurable constants ----
# You MUST set these to something reasonable for your test images.

# Example: there is a known object (e.g., a fork of length 18 cm) in the image.
# Measure its length in pixels using an image viewer, and fill in here.
REF_LENGTH_PIXELS = 300   # TODO: change this!
REF_LENGTH_CM = 18.0      # TODO: change this!

# Rough density (g/cm^3) and kcal per 100g for your food.
# Example: pizza
FOOD_DENSITY_G_PER_CM3 = 0.8
KCAL_PER_100G = 266

def create_color_overlay(img_bgr, mask_binary):
    """
    Create a simple overlay: food regions in green with some transparency.
    mask_binary: uint8 0/255
    """
    overlay = img_bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)  # green
    alpha = 0.4

    mask_bool = mask_binary > 0
    overlay[mask_bool] = (alpha * color +
                          (1 - alpha) * overlay[mask_bool]).astype(np.uint8)
    return overlay


def main():
    # Load classifier
    model_path = "bdr_model.pkl"
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"{model_path} not found. Please run train_bdr.py first."
        )
    clf = joblib.load(model_path)
    print(f"[INFO] Loaded classifier from {model_path}")

    test_img_dir = os.path.join("datasets", "test_images")
    out_dir = os.path.join("datasets", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Collect test images
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_paths.extend(glob.glob(os.path.join(test_img_dir, ext)))
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        raise RuntimeError(f"No test images found in {test_img_dir}")

    print(f"[INFO] Found {len(img_paths)} test images.")

    # Prepare CSV file to save results
    csv_path = os.path.join(out_dir, "results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "filename",
        "area_pixels",
        "volume_cm3",
        "mass_g",
        "kcal_est",
    ])

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        print(f"[INFO] Processing {img_name} ...")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] Cannot read {img_path}, skip.")
            continue

        # Run BDR segmentation
        label_map = segment_image_dct_bayes(img_bgr, clf,
                                            patch_size=8, stride=8)

        # Convert label map to binary mask:
        # Here we simply say "anything > 0 is food"
        mask_binary = (label_map > 0).astype(np.uint8) * 255

        # Estimate calories
        kcal_est, volume_cm3, mass_g = estimate_calories(
            mask_binary,
            ref_length_pixels=REF_LENGTH_PIXELS,
            ref_length_cm=REF_LENGTH_CM,
            food_density_g_per_cm3=FOOD_DENSITY_G_PER_CM3,
            kcal_per_100g=KCAL_PER_100G
        )

        area_pixels = int(np.count_nonzero(mask_binary > 0))
        print(f"    area_pixels={area_pixels}, "
              f"volume_cm3={volume_cm3:.2f}, "
              f"mass_g={mass_g:.2f}, "
              f"kcal={kcal_est:.2f}")

        # Save mask & overlay
        stem, _ = os.path.splitext(img_name)
        mask_path = os.path.join(out_dir, f"{stem}_mask.png")
        overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")

        cv2.imwrite(mask_path, mask_binary)
        overlay = create_color_overlay(img_bgr, mask_binary)
        cv2.imwrite(overlay_path, overlay)

        # Write to CSV
        csv_writer.writerow([
            img_name,
            area_pixels,
            f"{volume_cm3:.4f}",
            f"{mass_g:.4f}",
            f"{kcal_est:.4f}",
        ])

    csv_file.close()
    print(f"[INFO] Saved results to {csv_path}")


if __name__ == "__main__":
    main()
