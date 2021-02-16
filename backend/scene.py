import math
import random
from enum import Enum, auto

import numpy as np

from backend.utils import colors, rectangle, triangle, circle

MIN_SIZE_NOISE = 40
MAX_SIZE_NOISE = 100

MIN_SIZE_SHAPE = 1000
MAX_SIZE_SHAPE = 2500

MIN_VELOCITY = 1
MAX_VELOCITY = 3


def random_color():
    return [int(x) for x in colors[random.randint(0, colors.shape[0] - 1)]]


class Shape(Enum):
    RECTANGLE = auto()
    TRIANGLE = auto()
    CIRCLE = auto()


class SObject:
    def __init__(self, shape_type, **params):
        self.shape_type = shape_type
        self.params = params
        self.velocity = np.array([random.randint(MIN_VELOCITY, MAX_VELOCITY),
                                  random.randint(MIN_VELOCITY, MAX_VELOCITY)])
        self.recompute()

    def get_keypoints(self):
        if self.shape_type == Shape.CIRCLE:
            return ['centerPt']
        if self.shape_type == Shape.RECTANGLE:
            return ['pt1', 'pt2']
        return ['pt1', 'pt2', 'pt3']

    def get_points(self):
        return {x : self.params[x] for x in self.get_keypoints()}

    def recompute(self):
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
    def __init__(self, shape=(), num_noise=random.choice(range(16, 23)), num_objects=random.choice(range(4, 7))):
        self.shape = shape
        self.img = (np.ones(shape + (3,)) * 255).astype(np.uint8)
        self.num_noise = num_noise
        self.num_objects = num_objects
        self.noises : list[SObject] = []
        self.objects: list[SObject] = []

        self.generate(noise=True)
        self.generate(noise=False)

        self.rectify_collisions()

    def rectify_collisions(self):
        l = self.objects.copy()
        for o in l:
            d = o.get_points()
            for k, (x, y) in d.items():
                o.params[k] = (x + max(0, -o.boundMinX) - max(0, o.boundMaxX - self.img.shape[1]),
                                y + max(0, -o.boundMinY) - max(0, o.boundMaxY - self.img.shape[0]))
            o.recompute()
        while len(l):
            o1 = l.pop()
            for o2 in l:
                colliding = o1.boundMinY - o1.boundMaxX

    def render(self):
        self.img = (np.ones(self.shape + (3,)) * 255).astype(np.uint8)
        for noise in self.noises:
            if noise.shape_type == Shape.CIRCLE:
                circle(self.img, **noise.params)
            elif noise.shape_type == Shape.RECTANGLE:
                rectangle(self.img, **noise.params)
            else:
                triangle(self.img, **noise.params)
        for o in self.objects:
            if o.shape_type == Shape.CIRCLE:
                circle(self.img, **o.params)
            elif o.shape_type == Shape.RECTANGLE:
                rectangle(self.img, **o.params)
            else:
                triangle(self.img, **o.params)

    def generate(self, noise=False):
        size = self.num_noise if noise else self.num_objects
        obj_x = np.random.randint(self.img.shape[0], size=(size,))
        obj_y = np.random.randint(self.img.shape[1], size=(size,))
        obj_xy = np.column_stack([obj_x, obj_y])
        shapes = list(Shape)

        # Generating noise
        for x, y in obj_xy:
            shape = random.choice(shapes)
            if shape == Shape.CIRCLE:
                self.generate_circle((x, y), noise=noise)

            elif shape == Shape.RECTANGLE:
                self.generate_rectangle((x, y), noise=noise)

            else:
                self.generate_triangle((x, y), noise=noise)

    def generate_circle(self, pos, noise=False):
        x, y = pos
        a, b = (MIN_SIZE_NOISE, MAX_SIZE_NOISE) if noise else (MIN_SIZE_SHAPE, MAX_SIZE_SHAPE)
        params = {
            'centerPt': (y, x),
            'radius': round(random.uniform(math.sqrt(a / math.pi), math.sqrt(b / math.pi))),
            'color': random_color(),
        }
        circle(self.img, **params)
        o = SObject(Shape.CIRCLE, **params)
        if noise:
            self.noises.append(o)
        else:
            self.objects.append(o)

    def generate_rectangle(self, pos, noise=False):
        x, y = pos
        u, v = (MIN_SIZE_NOISE, MAX_SIZE_NOISE) if noise else (MIN_SIZE_SHAPE, MAX_SIZE_SHAPE)
        a = random.randint(int(math.sqrt(u)), int(math.sqrt(v)))
        b = random.randint(round(u / a), round(v / a))
        params = {
            'pt1': (y, x),
            'pt2': (y + b, x + a),
            'color': random_color(),
        }
        rectangle(self.img, **params)
        o = SObject(Shape.RECTANGLE, **params)
        if noise:
            self.noises.append(o)
        else:
            self.objects.append(o)

    def generate_triangle(self, pos, noise=False):
        x, y = pos
        w, z = (MIN_SIZE_NOISE, MAX_SIZE_NOISE) if noise else (MIN_SIZE_SHAPE, MAX_SIZE_SHAPE)
        u = int(math.sqrt(4 * w / math.sqrt(3)))
        v = int(math.sqrt(4 * z / math.sqrt(3)))
        a = random.randint(u, v)
        pt1 = (y, x)
        pt2 = (y + a, x)
        pt3 = (y + round(a / 2), np.clip(x - a, 0, self.img.shape[1]))
        params = {
            'pt1': pt1,
            'pt2': pt2,
            'pt3': pt3,
            'color': random_color(),
        }
        triangle(self.img, **params)
        o = SObject(Shape.TRIANGLE, **params)
        if noise:
            self.noises.append(o)
        else:
            self.objects.append(o)

    def collisions(self):
        for i in range(self.objects):
            for j in range(i+1, self.objects):
                if (self.objects[i].boundMaxX >= self.objects[j].boundMinX and \
                    self.objects[i].boundMaxX <= self.objects[j].boundMaxX) or \
                    (self.objects[i].boundMaxX <= self.objects[j].boundmaxX and \
                    self.objects[i].boundMinX >= self.objects[j].boundMinX) or \
                    (self.objects[i].boundMaxX >= self.objects[j].boundmaxX and \
                    self.objects[i].boundMinX <= self.objects[j].boundMinX):
                    
                    self.objects[j].velocity[0] = (-1 * self.objects[j].velocity[0] + self.objects[i].velocity[0]) / 2
                    self.objects[i].velocity[0] = (-1 * self.objects[i].velocity[0] + self.objects[j].velocity[0]) / 2
                    
                elif (self.objects[i].boundMaxY >= self.objects[j].boundMinY and \
                    self.objects[i].boundMaxY <= self.objects[j].boundMaxY) or \
                    (self.objects[i].boundMaxY <= self.objects[j].boundmaxY and \
                    self.objects[i].boundMinY >= self.objects[j].boundMinY) or \
                    (self.objects[i].boundMaxY >= self.objects[j].boundmaxY and \
                    self.objects[i].boundMinY <= self.objects[j].boundMinY):

                    self.objects[j].velocity[1] = (-1 * self.objects[j].velocity[1] + self.objects[i].velocity[1]) / 2
                    self.objects[i].velocity[1] = (-1 * self.objects[i].velocity[1] + self.objects[j].velocity[1]) / 2
                    
        for obj in self.objects:
            if obj.boundMinX == 0 or obj.boundMaxX == self.img.shape(0) - 1:
                obj.velocity[0] *= -1
            elif obj.boundMinY == 0 or obj.boundMaxY == self.img.shape(1) - 1:
                obj.velocity[1] *= -1
           
            
        
    
    def frame(self):
        self.collisions()
        for obj in self.objects:
            if shape == Shape.CIRCLE:
                obj.params['centerPt'] = (obj.velocity[0]+obj.params['centerPt'][0], obj.velocity[1]+obj.params['centerPt'][1])
            elif shape == Shape.RECTANGLE:
                obj.params['pt1'] = (obj.velocity[0]+obj.params['pt1'][0], obj.velocity[1]+obj.params['pt1'][1])
                obj.params['pt2'] = (obj.velocity[0]+obj.params['pt2'][0], obj.velocity[1]+obj.params['pt2'][1])
            else:
                obj.params['pt1'] = (obj.velocity[0]+obj.params['pt1'][0], obj.velocity[1]+obj.params['pt1'][1])
                obj.params['pt2'] = (obj.velocity[0]+obj.params['pt2'][0], obj.velocity[1]+obj.params['pt2'][1])
                obj.params['pt3'] = (obj.velocity[0]+obj.params['pt3'][0], obj.velocity[1]+obj.params['pt3'][1])
                



