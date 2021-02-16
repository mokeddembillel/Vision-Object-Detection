import cv2
import numpy as np


def mean_filter(path, kernel_size=3):
    img = cv2.imread(path)
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / kernel.size
    img = cv2.filter2D(img, 3, kernel)
    return img

def median_filter(path, kernel_size=3):
    img = cv2.imread(path)
    return cv2.medianBlur(img, kernel_size)

def gaussian_filter(path, kernel_size=3, sigma=0.5):
    img = cv2.imread(path)
    img = cv2.GaussianBlur(img, kernel_size, sigmaX=sigma)
    return img

def laplacian_filter(path, kernel_size=3):
    img = cv2.imread(path)
    img = cv2.Laplacian(img, 3, ksize=kernel_size)
    return img

def erode(path, kernel_size=3):
    img = cv2.imread(path)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def dilation(path, kernel_size=3):
    img = cv2.imread(path)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def gradient(path, kernel_size=3):
    img = cv2.imread(path)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return img