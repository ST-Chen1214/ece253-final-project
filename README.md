# ece253-final-project
UCSD ECE-253 Project
# ECE 253 Final Project  
## Food Image Preprocessing and Calorie Estimation Pipeline

---

## Project Overview
This project presents an end-to-end pipeline for **food calorie estimation from images**, developed as the final project for **ECE 253 – Digital Image Processing**.  
The pipeline integrates **classical computer vision**, **learning-based image enhancement**, and **probabilistic modeling** to study how different preprocessing strategies affect downstream calorie estimation performance.

The overall workflow consists of **three main stages**, each implemented in a separate module:

1. Image pre-processing using traditional camera calibration methods  
2. Image pre-processing using a deep learning–based distortion correction model  
3. Final calorie estimation using Bayesian Density Regression (BDR)

---

## Repository Structure
The repository is organized into the following three main directories:

```text
project/
├─ traditional/     # Classical image pre-processing (camera calibration & distortion correction)
├─ udcnet/          # Learning-based image pre-processing (deep geometric correction)
└─ bdrNestmtn/      # Final stage: Bayesian Density Regression for calorie estimation
```