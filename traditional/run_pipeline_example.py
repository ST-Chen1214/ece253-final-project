# === file: run_pipeline_example.py ===
import numpy as np
import cv2

from classical_pipeline import classical_full_pipeline

def main():
    # 1. Load camera parameters (if you want rectification)
    try:
        params = np.load("camera_params.npz")
        camera_matrix = params["camera_matrix"]
        dist_coeffs = params["dist_coeffs"]
        print("[INFO] Loaded camera parameters.")
    except FileNotFoundError:
        print("[WARN] camera_params.npz not found. "
              "Proceeding WITHOUT geometric rectification.")
        camera_matrix = None
        dist_coeffs = None

    # 2. Load test image
    img_path = "test_food.jpg"  # change if needed
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image at {img_path}")

    # 3. Run classical pipeline
    result = classical_full_pipeline(img_bgr,
                                     camera_matrix=camera_matrix,
                                     dist_coeffs=dist_coeffs)

    # 4. Show before/after
    cv2.imshow("Original", img_bgr)
    cv2.imshow("Rectified + Illumination Enhanced", result)
    print("[INFO] Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5. Save result to file
    cv2.imwrite("test_food_processed.jpg", result)
    print("[INFO] Saved processed image to test_food_processed.jpg")

if __name__ == "__main__":
    main()
