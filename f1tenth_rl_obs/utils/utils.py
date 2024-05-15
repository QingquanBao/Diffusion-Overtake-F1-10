import numpy as np
from scipy.interpolate import CubicSpline
from numba import njit
from numba.np.extensions import cross2d
import math
import random

################## GLOBAL DIRECTORY VARIABLES ##################
WORKDIR = "/Users/mac/Desktop/PENN/f1tenth_rl_obs/"
RAYLOGDIR = WORKDIR + "ray_rl/ray_results"
################## GLOBAL DIRECTORY VARIABLES ##################


class PushOnlyCircularQueue:
    def __init__(self, q_size):
        self.q_size = q_size
        self.queue = [0.0]*q_size
        self.front_pt = 0
        # Point to the next position in the buffer to be inserted
        self.rear_pt = 0
        self.is_full = False
        self.size = 0

    def push(self, item):
        if self.is_full:
            self.front_pt = (self.front_pt + 1) % self.q_size
            self.size = self.q_size
        else:
            self.size += 1
        self.queue[self.rear_pt] = item
        self.rear_pt = (self.rear_pt + 1) % self.q_size
        self.is_full = (self.rear_pt == self.front_pt)

    def get_front(self):
        return self.queue[self.front_pt]

    def get_rear(self):
        return self.queue[self.rear_pt-1]

    def get_diff(self):
        return self.queue[self.rear_pt-1] - self.queue[self.front_pt]
