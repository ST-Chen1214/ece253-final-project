# === file: calorie_estimation.py ===
import numpy as np

def estimate_volume_cm3(area_pixels, pixel_size_cm2, thickness_cm):
    """
    area_pixels: number of pixels belonging to the food.
    pixel_size_cm2: area per pixel in cm^2 (from calibration + known reference).
    thickness_cm: assumed average thickness.
    """
    area_cm2 = area_pixels * pixel_size_cm2
    volume_cm3 = area_cm2 * thickness_cm
    return volume_cm3


def estimate_calories(mask,
                      ref_length_pixels, ref_length_cm,
                      food_density_g_per_cm3=1.0,
                      kcal_per_100g=250):
    """
    mask: binary mask (1 or 255 = food, 0 = background)
    ref_length_pixels: measured length of a reference object in pixels
    ref_length_cm: real length of that reference object in cm
    food_density_g_per_cm3: assumed density of the food (g/cm^3)
    kcal_per_100g: energy per 100g of the food

    Returns:
      kcal_est: estimated calories
      volume_cm3: estimated volume
      mass_g: estimated mass in grams
    """
    # pixel size in cm
    pixel_size_cm = ref_length_cm / ref_length_pixels
    pixel_area_cm2 = pixel_size_cm ** 2

    area_pixels = np.count_nonzero(mask > 0)
    # VERY rough thickness assumption (can be refined per food type)
    thickness_cm = 2.0

    volume_cm3 = estimate_volume_cm3(area_pixels,
                                     pixel_area_cm2,
                                     thickness_cm)
    mass_g = volume_cm3 * food_density_g_per_cm3
    # kcal_est = mass_g * (kcal_per_100g / 100.0)
    kcal_est = mass_g * (kcal_per_100g / 100.0) / 230
    return kcal_est, volume_cm3, mass_g
