import numpy as np
from random import randint
import cv2



def rectangle(img, pt1, pt2, color, fill):
    cv2.rectangle(img, pt1, pt2, color, fill)

def circle(img, centerPt, radius, color, fill):
    cv2.circle(img, centerPt, radius, color, fill)

def triangle(img, pt1, pt2, pt3, color):
    pts = np.array([pt1, pt2, pt3], np.int32)
    cv2.fillPoly(img,[pts], color)

def generateObjects(img, type, width, height, minPt, maxPt):
    if type == 'rect':
        pt1 = (randint(minPt, maxPt), randint(minPt, maxPt))
        pt2 = (pt1[0] + height, pt1[1] + width)

    elif type == 'circle':

        

    elif type == 'triangle':





colors = {
    'red': (247, 15, 2),
    'brown': (145, 73, 0),
    'orange': (247, 99, 0),
    'yellow': (247, 246, 3),
    'pink': (247, 59, 173),
    'purple': (99, 49, 147),
    'blue': (28, 5, 243),
    'cyan': (0, 246, 246),
    'green': (6, 247, 1),
    'black': (0, 0, 0)
}

objects = []

def createVideo(objects):
    img = cv2.imread('data\\billel.jpg', 1)
    img = cv2.resize(img, (960, 540)) 

    rectangle(img, (0,0), (255,255), (155, 255, 0), -1)
    circle(img, (400, 400), 50, (155, 255, 0), -1)
    triangle(img, [100,100],[100,200],[200,150], (255,255,0))

    

    pts = np.array([[100,100],[100,200],[200,150]], np.int64)
    cv2.fillPoly(img,[pts],(255,255,0))


    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.distroyAllWindows(0)

createVideo(objects)    








# def createVideo(objects):
#     img = cv2.imread('data\\billel.jpg', 1)
#     img = cv2.resize(img, (960, 540))            
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     cv2.distroyAllWindows()

# createVideo(objects)    