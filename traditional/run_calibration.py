# === file: run_calibration.py ===
import glob
import numpy as np
import cv2

from classical_pipeline import calibrate_camera_from_checkerboard

def main():
    # 1. Collect all checkerboard images
    calib_folder = "./calib_images"  # change if needed
    image_paths = glob.glob(calib_folder + "/*.jpg")
    image_paths += glob.glob(calib_folder + "/*.png")

    print("[INFO] Found", len(image_paths), "calibration images.")
    if len(image_paths) == 0:
        raise RuntimeError("No calibration images found. "
                           "Put them in ./calib_images")

    # 2. Calibrate
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_from_checkerboard(
        image_paths,
        board_size=(9, 6),   # change if your board is different
        square_size=1.0      # e.g., 1.0cm; relative scale is enough if you only want rectification
    )

    print("[INFO] Camera matrix:\n", camera_matrix)
    print("[INFO] Distortion coeffs:\n", dist_coeffs)

    # 3. Save parameters to file for later use
    np.savez("camera_params.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)

    print("[INFO] Saved camera_params.npz")


if __name__ == "__main__":
    main()
