import cv2
import numpy as np


def mean_filter(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / kernel.size
    img = cv2.filter2D(img, 3, kernel).astype(np.uint8)
    return img


def median_filter(img, kernel_size=3):
    return cv2.medianBlur(img, kernel_size)


def gaussian_filter(img, kernel_size=3, sigma=1):
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)
    return img


def laplacian(img, kernel_size=3):
    img = cv2.Laplacian(img, 3, ksize=kernel_size)
    img -= img.min()
    img = img / img.max() * 255
    img = img.astype(np.uint8)
    return img


def erosion(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img


def dilation(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def gradient(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return img
