import math
import random
from enum import Enum, auto

import numpy as np

from backend.utils import rectangle, triangle, circle, cosinus, perpendicular, minimum_distance
from backend.colors import colors

MIN_SIZE_NOISE = 40
MAX_SIZE_NOISE = 100

MIN_SIZE_SHAPE = 1000
MAX_SIZE_SHAPE = 2500

MIN_VELOCITY = 2
MAX_VELOCITY = 6

UPDATE_NOISE_DELAY = 60

SCENE = None


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
        self.is_colliding = False

    def get_keypoints(self):
        if self.shape_type == Shape.CIRCLE:
            return ['centerPt']
        if self.shape_type == Shape.RECTANGLE:
            return ['pt1', 'pt2']
        return ['pt1', 'pt2', 'pt3']

    def get_points(self):
        return {x: self.params[x] for x in self.get_keypoints()}

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

    def is_colliding_with_object(self, o):
        if self.shape_type == Shape.CIRCLE and o.shape_type == Shape.CIRCLE:
            r1, r2 = self.params['radius'], o.params['radius']
            c1, c2 = np.array(self.params['centerPt']), np.array(o.params['centerPt'])
            d = c2 - c1
            norm_d = np.linalg.norm(d)
            u = d / norm_d if norm_d != 0 else 0
            u = c1 + u * r1
            return norm_d < r1 + r2, u
        if self.shape_type == Shape.CIRCLE and o.shape_type == Shape.RECTANGLE:
            def intersect_circle(center, radius, A, B):
                d, p = minimum_distance(A, B, center)
                return d < radius, p

            A, C = np.array(o.params['pt1']), np.array(o.params['pt2'])
            w, h = C - A
            B, D = np.array([0, h]) + A, np.array([w, 0]) + A
            P, r = np.array(self.params['centerPt']), self.params['radius']
            if 0 <= (P - A) @ (B - A) <= (B - A) @ (B - A) and 0 <= (P - A) @ (D - A) <= (D - A) @ (D - A):
                return True, o.cog()
            for X, Y in [(A, B), (B, C), (C, D), (D, A)]:
                c, p = intersect_circle(P, r, X, Y)
                if c:
                    return c, p
            return False, None

        if self.shape_type in [Shape.RECTANGLE, Shape.TRIANGLE] and o.shape_type in [Shape.RECTANGLE, Shape.TRIANGLE]:
            if self.boundMaxX < o.boundMinX or o.boundMaxX < self.boundMinX or \
                    self.boundMaxY < o.boundMinY or o.boundMaxY < self.boundMinY:
                return False, None

            return True, (self.cog() - o.cog()) / 2

        if self.shape_type == Shape.TRIANGLE and o.shape_type == Shape.CIRCLE:
            def intersect_circle(center, radius, A, B):
                d, p = minimum_distance(A, B, center)
                return d < radius, p

            A, B, C = np.array(list(self.get_points().values()))
            P, r = np.array(o.params['centerPt']), o.params['radius']
            if 0 <= (P - A) @ (B - A) <= (B - A) @ (B - A) and 0 <= (P - A) @ (C - A) <= (C - A) @ (C - A):
                return True, o.cog()
            for X, Y in [(A, B), (B, C), (C, A)]:
                c, p = intersect_circle(P, r, X, Y)
                if c:
                    return c, p
            return False, None

        return o.is_colliding_with_object(self)

    def is_colliding_with(self):
        try:
            l = [(x, self.is_colliding_with_object(x)) for x in SCENE.objects if x is not self]
            l = [x for x in l if x[1][0]][0]
            return l
        except IndexError:
            return None, (False, None)

    def velocity_change_collision(self, u, coef=1):
        self.velocity = -self.velocity + 2 * u * np.linalg.norm(self.velocity) / np.linalg.norm(u) * cosinus(
            self.velocity, u)
        self.velocity = coef * np.ceil(self.velocity).astype(np.int)

    def cog(self):
        return np.array(list(self.get_points().values())).mean(0)


class Scene:
    def __init__(self, shape=(), num_noise=random.choice(range(16, 23)), num_objects=random.choice(range(4, 7)),
                 empty=False, shape_types=list(Shape)):
        self.shape = shape
        self.img = (np.ones(shape + (3,)) * 255).astype(np.uint8)
        self.num_noise = num_noise
        self.num_objects = num_objects
        self.noises: list[SObject] = []
        self.objects: list[SObject] = []
        self.x_unit = np.array([0, 1])
        self.y_unit = np.array([1, 0])
        global SCENE
        SCENE = self
        self.shape_types = shape_types

        if not empty:
            self.generate(noise=True)
            self.generate(noise=False)

    def rectify_boundary_collisions(self):
        l = self.objects.copy()
        for o in l:
            d = o.get_points()
            for k, (x, y) in d.items():
                o.params[k] = (x + max(0, -o.boundMinX) - max(0, o.boundMaxX - self.img.shape[1]),
                               y + max(0, -o.boundMinY) - max(0, o.boundMaxY - self.img.shape[0]))
            o.recompute()

    def rectify_object_collisions(self):
        l = self.objects.copy()
        while len(l):
            o1 = l.pop()
            for o2 in self.objects:
                a = max(0, o1.boundMaxX - o2.boundMinX)
                if o2 is o1 or o2.boundMaxX < o1.boundMinX or a == 0:
                    continue
                for k, (x, y) in o2.get_points().items():
                    o2.params[k] = (x + a, y)
                o2.recompute()
                # l.append(o2)

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

        if noise:
            pcolors = colors[np.random.randint(0, colors.shape[0], (size, ))]
        else:
            pcolors = random.sample([c for c in colors], size)

        # Generating noise
        for i, (x, y) in enumerate(obj_xy):
            c = [int(x) for x in pcolors[i]][::-1]
            shape = random.choice(self.shape_types)
            if shape == Shape.CIRCLE:
                self.generate_circle((x, y), noise=noise, color=c)

            elif shape == Shape.RECTANGLE:
                self.generate_rectangle((x, y), noise=noise, color=c)

            else:
                self.generate_triangle((x, y), noise=noise, color=c)

    def generate_circle(self, pos, noise=False, color=None):
        x, y = pos
        a, b = (MIN_SIZE_NOISE, MAX_SIZE_NOISE) if noise else (MIN_SIZE_SHAPE, MAX_SIZE_SHAPE)
        params = {
            'centerPt': (y, x),
            'radius': round(random.uniform(math.sqrt(a / math.pi), math.sqrt(b / math.pi))),
            'color': color,
        }
        circle(self.img, **params)
        o = SObject(Shape.CIRCLE, **params)
        if noise:
            self.noises.append(o)
        else:
            self.objects.append(o)

    def generate_rectangle(self, pos, noise=False, color=None):
        x, y = pos
        u, v = (MIN_SIZE_NOISE, MAX_SIZE_NOISE) if noise else (MIN_SIZE_SHAPE, MAX_SIZE_SHAPE)
        a = random.randint(int(math.sqrt(u)), int(math.sqrt(v)))
        b = random.randint(round(u / a), round(v / a))
        params = {
            'pt1': (y, x),
            'pt2': (y + b, x + a),
            'color': color,
        }
        rectangle(self.img, **params)
        o = SObject(Shape.RECTANGLE, **params)
        if noise:
            self.noises.append(o)
        else:
            self.objects.append(o)

    def generate_triangle(self, pos, noise=False, color=None):
        x, y = pos
        w, z = (MIN_SIZE_NOISE, MAX_SIZE_NOISE) if noise else (MIN_SIZE_SHAPE, MAX_SIZE_SHAPE)
        u = int(math.sqrt(4 * w / math.sqrt(3)))
        v = int(math.sqrt(4 * z / math.sqrt(3)))
        a = random.randint(u, v)
        pt1 = (y, x)
        pt2 = (y + a, x)
        pt3 = (y + round(a / 2), x - a)
        params = {
            'pt1': pt1,
            'pt2': pt2,
            'pt3': pt3,
            'color': color,
        }
        triangle(self.img, **params)
        o = SObject(Shape.TRIANGLE, **params)
        if noise:
            self.noises.append(o)
        else:
            self.objects.append(o)

    def collisions(self):
        for o in self.objects:
            oo, (o_colliding, pos) = o.is_colliding_with()
            if o.boundMinX <= 0 or o.boundMaxX >= self.img.shape[1] - 1:
                if not o.is_colliding:
                    o.velocity_change_collision(self.x_unit)
                    o.is_colliding = True

            elif o.boundMinY <= 0 or o.boundMaxY >= self.img.shape[0] - 1:
                if not o.is_colliding:
                    o.velocity_change_collision(self.y_unit)
                    o.is_colliding = True

            elif o_colliding:
                if not o.is_colliding:
                    if o.shape_type in [Shape.RECTANGLE, Shape.TRIANGLE]:
                        pos = max(self.x_unit, self.y_unit, key=lambda x: cosinus(x, pos))
                    else:
                        pos = np.array(o.cog()) - pos
                    cos = 1 if cosinus(o.velocity, pos) >= 0 else -1
                    o.velocity_change_collision(perpendicular(pos), -cos)
                    o.is_colliding = True

            else:
                o.is_colliding = False

            if o.boundMaxX <= 0 or o.boundMinX >= self.img.shape[1] - 1 or o.boundMaxY <= 0 or o.boundMinY >= \
                    self.img.shape[0] - 1:
                center = (np.array(self.shape) / 2).astype(np.int)[::-1]
                cog = o.cog()
                d = center - cog
                d = d / np.linalg.norm(d) * np.linalg.norm(o.velocity)
                o.velocity = np.ceil(d).astype(np.int)

    def frame(self):
        self.collisions()
        for obj in self.objects:
            if obj.shape_type == Shape.CIRCLE:
                obj.params['centerPt'] = (
                    obj.velocity[0] + obj.params['centerPt'][0], obj.velocity[1] + obj.params['centerPt'][1])
            elif obj.shape_type == Shape.RECTANGLE:
                obj.params['pt1'] = (obj.velocity[0] + obj.params['pt1'][0], obj.velocity[1] + obj.params['pt1'][1])
                obj.params['pt2'] = (obj.velocity[0] + obj.params['pt2'][0], obj.velocity[1] + obj.params['pt2'][1])
            else:
                obj.params['pt1'] = (obj.velocity[0] + obj.params['pt1'][0], obj.velocity[1] + obj.params['pt1'][1])
                obj.params['pt2'] = (obj.velocity[0] + obj.params['pt2'][0], obj.velocity[1] + obj.params['pt2'][1])
                obj.params['pt3'] = (obj.velocity[0] + obj.params['pt3'][0], obj.velocity[1] + obj.params['pt3'][1])
            obj.recompute()
