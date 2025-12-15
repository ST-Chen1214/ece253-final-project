# ECE 253 Final Project  
## Traditional Camera Calibration and Geometric Correction Pipeline

---

## Overview
This project implements a **traditional camera calibration and geometric distortion correction pipeline**.
The goal is to correct lens-induced geometric distortions using classical computer vision techniques,
and to provide a baseline for comparison with learning-based methods.

---

## Directory Structure
Please ensure the following structure is used for the traditional pipeline:

```text
traditional/
├─ calib_images/ # Calibration images (e.g., checkerboard photos)
├─ camera_params.npz # Generated after calibration
├─ classical_pipeline.py
├─ run_calibration.py
├─ run_pipeline_example.py
├─ test_food.jpg # Input test image (must be in this directory)
└─ test_food_processed.jpg # Output image (generated after running pipeline)
```
---

## Environment Requirements
- Python 3.8 or later

## Run Camera Calibration
```bash
python run_calibration.py
python run_pipeline_example.py
```