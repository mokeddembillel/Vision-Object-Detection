from collections import defaultdict

import cv2
import numpy as np

from backend.colors import rgb2color, colors

num_edges2shape = {
    3 : 'triangle',
    4 : 'rectangle',
}
num_edges2shape = defaultdict(lambda:'circle', num_edges2shape)

def mean_filter(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / kernel.size
    img = cv2.filter2D(img, 3, kernel).astype(np.uint8)
    return img

def gaussian_filter(img, kernel_size=3, sigma=1):
    kernel = [1/(sigma*np.sqrt(np.pi*2))*np.exp(-0.5*(x/sigma)**2) for x in range(-int((kernel_size-1)/2), int((kernel_size+1)/2))]
    kernel = np.array(kernel)
    kernel = kernel.reshape(kernel_size,1) @ kernel.reshape(1, kernel_size)
    kernel = kernel / kernel.sum()
    img = cv2.filter2D(img, 3, kernel).round().astype(np.uint8)
    return img

def laplacian(img, kernel_size=3):
    c = int((kernel_size - 1)/2)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.int)
    kernel[c, c] = - kernel.sum() + 1
    img = cv2.filter2D(img, 3, kernel)
    img -= img.min()
    img = img / img.max() * 255
    img = img.astype(np.uint8)
    return img

def median_filter(img, kernel_size=3):
    return cv2.medianBlur(img, kernel_size)

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
    if len(x.shape) == 1:
        return 0
    return 0.5*np.abs(np.dot(x[:,0],np.roll(x[:,1],1))-np.dot(x[:,1],np.roll(x[:,0],1)))

def detect(img, area_th=500):
    _, th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 240, 255, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if poly_area(np.squeeze(c)) > area_th]
    results = []
    for contour in contours:
        num_edges = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True).shape[0]
        # num_edges = number_of_edges(contour, 1.12)
        shape = num_edges2shape[num_edges]
        cog = contour.squeeze().mean(0).astype(np.int)
        color = img[cog[1], cog[0]]
        color = ((colors - color[::-1])**2).sum(-1).argmin()
        color = colors[color]
        color = rgb2color[tuple(color)]
        results.append((cog, shape, color))
    return results

def number_of_edges(contour, coef=1.12):
    contour = contour.squeeze()
    num_edges = 0
    contour = [contour[i] for i in range(contour.shape[0])]
    current = []
    arc = 0
    for c in contour:
        current.append(c)
        if len(current) < 2:
            continue
        arc += np.linalg.norm(current[-1] - current[-2])
        d = np.linalg.norm(c - current[0])
        if arc > d*coef:
            num_edges += 1
            current = []
            arc = 0
    return num_edges + 1


def cosinus(v1, v2):
    s = np.sqrt((v1 ** 2).sum() * (v2 ** 2).sum())
    if s == 0:
        return 0
    return (v1 @ v2) / np.sqrt((v1**2).sum() * (v2**2).sum())

def perpendicular(u):
    if u[0] == 0:
        return np.array([1, 0])
    elif u[1] == 0:
        return np.array([0, 1])
    a = np.zeros(u.shape)
    a[0] = 1
    a[1] = -u[0]/u[1]
    return a


def rectangle(img, pt1, pt2, color):
    cv2.rectangle(img, pt1, pt2, color, -1)

def circle(img, centerPt, radius, color):
    cv2.circle(img, centerPt, radius, color, -1)

def triangle(img, pt1, pt2, pt3, color):
    pts = np.array([pt1, pt2, pt3], np.int32)
    cv2.fillPoly(img,[pts], color)

def equation_line(A, B):
    a = A - B
    a = a[1] / a[0]
    b = A[1] - a*A[0]
    return a, b

def minimum_distance(v, w, p):
    l2 = ((v - w)**2).sum()
    if l2 == 0:
        return np.linalg.norm(p - v)
    t = max(0, min(1, (p - v) @ (w - v) / l2))
    projection = v + t * (w - v)
    return np.linalg.norm(p - projection), projection
