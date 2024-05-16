# -*- coding: utf-8 -*-
"""Histogram_detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1W9UnFHh_SB2JqrA4eutOCeaqDLgRhpwj
"""

import cv2
import numpy as np

def detect_histogram_equalization(image_path, threshold=0.90):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    rows, cols = image.shape

    black_pixels = np.sum(image < 3)
    white_pixels = np.sum(image > 253)

    total_pixels = rows * cols
    black_ratio = black_pixels / total_pixels
    white_ratio = white_pixels / total_pixels

    print(f'Black pixels: {black_pixels}, White pixels: {white_pixels}')
    print(f'Black ratio: {black_ratio:.4f}, White ratio: {white_ratio:.4f}')

    if black_ratio > threshold or white_ratio > threshold:
        print("Noise detected")
        return True
    else:
        print("No significant salt-and-pepper noise detected")
        return False

image_path = 'TC/13.png'
detect_histogram_equalization(image_path)

