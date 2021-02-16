from collections import defaultdict

import cv2
import numpy as np

num_edges2shape = {
    3 : 'triangle',
    4 : 'rectangle',
}
num_edges2shape = defaultdict(lambda:'circle', num_edges2shape)

rgb2color = {(255, 0, 0) : 'RED',
             (165,42,42) : 'BROWN',
             (128,0,128) : 'PURPLE',
             (0,255,255) : 'CYAN',
             (0, 255, 0) : 'GREEN',
             (0, 0, 255) : 'BLUE',
             (255,255,0) : 'YELLOW',
             (0,0,0) : 'BLACK',
             (255,192,203) : 'PINK',
             (255,165,0) : 'ORANGE'}

colors = np.array(list(rgb2color.keys()))

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

def poly_area(x):
    return 0.5*np.abs(np.dot(x[:,0],np.roll(x[:,1],1))-np.dot(x[:,1],np.roll(x[:,0],1)))

def detect(img, area_th=500):
    _, th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 240, 255, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if poly_area(np.squeeze(c)) > area_th]
    results = []
    for contour in contours:
        num_edges = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True).shape[0]
        shape = num_edges2shape[num_edges]
        cog = contour.squeeze().mean(0).astype(np.int)
        color = img[cog[1], cog[0]]
        color = ((colors - color[::-1])**2).sum(-1).argmin()
        color = colors[color]
        color = rgb2color[tuple(color)]
        results.append((cog, shape, color))
    return results


def rectangle(img, pt1, pt2, color):
    cv2.rectangle(img, pt1, pt2, color, -1)

def circle(img, centerPt, radius, color):
    cv2.circle(img, centerPt, radius, color, -1)

def triangle(img, pt1, pt2, pt3, color):
    pts = np.array([pt1, pt2, pt3], np.int32)
    cv2.fillPoly(img,[pts], color)
