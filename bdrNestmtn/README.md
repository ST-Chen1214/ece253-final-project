# ECE 253 Final Project  
## Bayesian Density Regression (BDR) for Food Calorie Estimation

---

## Overview
This project implements a **Bayesian Density Regression (BDR)** pipeline for estimating food calories
from images. The method leverages training images and corresponding segmentation masks to learn
a density-based regression model, and then applies the trained model to unseen test images to
estimate calorie-related quantities.

The final outputs include **CSV files with estimated values** and **mask overlay visualizations**
for qualitative inspection.

---

## Directory Structure
Please ensure the following directory structure is used before running the pipeline:

```text
bdrNestmtn/
├─ datasets/
│  ├─ train_images/     # Training food images
│  ├─ train_masks/      # Corresponding ground-truth masks for training images
│  ├─ test_images/      # Test food images for calorie estimation
│  └─ outputs/          # Generated CSV files and mask overlay results
├─ train_bdr.py         # Train Bayesian Density Regression model
└─ run_bdr_and_calories.py  # Run inference and calorie estimation on test images
```

---

## Environment Requirements
- Python 3.8 or later
- Required Python packages listed in the project environment (NumPy, OpenCV, etc.)

---

## Run BDR
```bash
python train_bdr.py
python run_bdr_and_calories.py
```