import numpy as np
from enum import Enum, auto
from backend.utils import colors

MIN_SIZE_NOISE = 2
MAX_SIZE_NOISE = 5

class Shape(Enum):
    RECTANGLE = auto()
    TRIANGLE = auto()
    CIRCLE = auto()


class SObject:
    def __init__(self, pos, shape_type, color, size):
        self.pos = pos
        self.shape_type = shape_type
        self.color = color
        self.size = size
        self.velocity = np.random.random((2,))

class Scene:
    def __init__(self, shape=(), num_noise=10):
        self.img = np.ones(*shape)
        self.num_noise = num_noise

    def generate_random_objects(self):
        noise_x = np.random.randint(self.img.shape[0], size=(self.num_noise,))
        noise_y = np.random.randint(self.img.shape[1], size=(self.num_noise,))
        noise = np.column_stack([noise_x, noise_y])


