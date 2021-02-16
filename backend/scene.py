import math
import random

import numpy as np
from enum import Enum, auto
from backend.utils import colors, rectangle, triangle, circle

MIN_SIZE_NOISE = 50
MAX_SIZE_NOISE = 200

class Shape(Enum):
    RECTANGLE = auto()
    TRIANGLE = auto()
    CIRCLE = auto()


class SObject:
    def __init__(self, shape_type, **params):
        self.shape_type = shape_type
        self.params = params
        self.velocity = np.random.random((2,))

        if self.shape_type == Shape.RECTANGLE:
            self.boundMinX = self.params['pt1'][0]
            self.boundMaxX = self.params['pt2'][0]
            self.boundMinY = self.params['pt1'][1]
            self.boundMaxY = self.params['pt2'][1]
        elif self.shape_type == Shape.TRIANGLE:
            self.boundMinX = min(self.params['pt1'][0], self.params['pt2'][0], self.params['pt3'][0])
            self.boundMaxX = max(self.params['pt1'][0], self.params['pt2'][0], self.params['pt3'][0])
            self.boundMinY = min(self.params['pt1'][1], self.params['pt2'][1], self.params['pt3'][1])
            self.boundMaxY = max(self.params['pt1'][1], self.params['pt2'][1], self.params['pt3'][1])
        else:
            self.boundMinX = self.params['centerPt'][0] - self.params['radius']
            self.boundMaxX = self.params['centerPt'][0] + self.params['radius']
            self.boundMinY = self.params['centerPt'][1] - self.params['radius']
            self.boundMaxY = self.params['centerPt'][1] + self.params['radius']

class Scene:
    def __init__(self, shape=(), num_noise=10):
        self.img = np.ones(*shape)
        self.num_noise = num_noise
        self.noises = []
        self.objects = []

        noise_x = np.random.randint(self.img.shape[0], size=(self.num_noise,))
        noise_y = np.random.randint(self.img.shape[1], size=(self.num_noise,))
        noise = np.column_stack([noise_x, noise_y])
        for x, y in noise:
            shape = random.choice(Shape)
            if shape == Shape.CIRCLE:
                params = {
                    'centerPt' : (y, x),
                    'radius' : random.uniform(math.sqrt(MIN_SIZE_NOISE/math.pi), math.sqrt(MAX_SIZE_NOISE/math.pi)),
                    'color' : colors[random.randint(0, len(colors.shape[0]))],
                    'shape' : shape
                }
                circle(self.img, **params)

            elif shape == Shape.RECTANGLE:
                params = {
                    'centerPt': (y, x),
                    'radius': random.uniform(math.sqrt(MIN_SIZE_NOISE / math.pi), math.sqrt(MAX_SIZE_NOISE / math.pi)),
                    'color': colors[random.randint(0, len(colors.shape[0]))],
                    'shape': shape
                }
                rectangle()





