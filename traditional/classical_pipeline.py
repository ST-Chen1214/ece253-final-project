# === file: classical_pipeline.py ===
import cv2
import numpy as np

#############################
# 1. Illumination (no-training)
#############################

def pixel_preprocess_lab(img_bgr):
    """
    Convert BGR -> L*a*b and apply simple brightness/contrast tweaks on L*.
    img_bgr: uint8 BGR image from cv2.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # CLAHE on L* channel for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_enh = clahe.apply(L)

    lab_enh = cv2.merge([L_enh, a, b])
    img_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    return img_enh


def local_preprocess_median(img_bgr, ksize=3):
    """
    Local pre-processing: non-linear median filter to suppress noise
    while preserving edges.
    """
    return cv2.medianBlur(img_bgr, ksize)


def full_classical_illumination(img_bgr):
    """
    Combined pixel + local pre-processing:
    1. L*a*b* CLAHE
    2. Median filter
    """
    img_pixel = pixel_preprocess_lab(img_bgr)
    img_local = local_preprocess_median(img_pixel, ksize=3)
    return img_local

#############################
# 2. Geometric (no-training): Calibration + Rectification
#############################

def calibrate_camera_from_checkerboard(image_paths,
                                       board_size=(9, 6),
                                       square_size=1.0):
    """
    Perform Zhang's calibration with a set of checkerboard images.
    board_size: (columns, rows) INNER corners
    square_size: physical square size (e.g. in cm) for scale.
    Returns: camera_matrix, dist_coeffs, rvecs, tvecs
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0],
                           0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARN] Cannot read image: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, board_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print(f"[WARN] Chessboard not found in: {fname}")

    if len(objpoints) == 0:
        raise RuntimeError("No valid checkerboard detections. "
                           "Check board_size and your images.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print("[INFO] Calibration reprojection error:", ret)
    return camera_matrix, dist_coeffs, rvecs, tvecs


def undistort_image(img_bgr, camera_matrix, dist_coeffs):
    """
    Geometric rectification using Brown–Conrady model via OpenCV's undistort.
    """
    h, w = img_bgr.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(img_bgr, camera_matrix, dist_coeffs, None,
                        new_camera_mtx)
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        dst = dst[y:y + h_roi, x:x + w_roi]
    return dst

#############################
# 3. Pipeline wrapper
#############################

def classical_full_pipeline(img_bgr, camera_matrix=None, dist_coeffs=None):
    """
    If camera_matrix and dist_coeffs are given, apply rectification first.
    Then do illumination enhancement.
    """
    if camera_matrix is not None and dist_coeffs is not None:
        img_bgr = undistort_image(img_bgr, camera_matrix, dist_coeffs)
    img_bgr = full_classical_illumination(img_bgr)
    return img_bgr
