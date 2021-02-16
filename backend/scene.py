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
                }
                circle(self.img, **params)
                self.noises.append(SObject(shape, **params))

            elif shape == Shape.RECTANGLE:
                a = random.randint(int(math.sqrt(MIN_SIZE_NOISE)), int(math.sqrt(MAX_SIZE_NOISE)))
                b = random.randint(round(MIN_SIZE_NOISE/a), round(MAX_SIZE_NOISE/a))
                params = {
                    'pt1': (y, x),
                    'pt2': (y + b, x + a),
                    'color': colors[random.randint(0, len(colors.shape[0]))],
                }
                rectangle(**params)
                self.noises.append(SObject(shape, **params))

            else:
                u = int(math.sqrt(4*MIN_SIZE_NOISE/math.sqrt(3)))
                v = int(math.sqrt(4*MAX_SIZE_NOISE/math.sqrt(3)))
                a = random.randint(u, v)
                b = random.randint()





