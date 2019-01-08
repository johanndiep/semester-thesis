# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file calculates a random direction for translation and rotation.


# external libraries
import random
import math
import numpy as np


# calculates random direction
class Randomizer():
    def __init__(self):
        # uniformly sampling inclination and azimuth angle
        self.theta_t = random.uniform(0, math.pi)
        self.theta_r = random.uniform(0, math.pi)
        self.phi_t = random.uniform(0, 2 * math.pi)
        self.phi_r = random.uniform(0, 2 * math.pi)

    def rand_position_offset(self, dist): 
        # calculating cartesian coordinates from spherical coordinates 
        x = dist * math.sin(self.theta_t) * math.cos(self.phi_t)
        y = dist * math.sin(self.theta_t) * math.sin(self.phi_t)
        z = dist * math.cos(self.theta_t)

        return np.array([x, y, z]) # return cartesian coordinates

    def rand_rotation_offset(self, angle):
        # calculating cartesian coordinates from spherical coordinates 
        a_x = math.sin(self.theta_r) * math.cos(self.phi_r)
        a_y = math.sin(self.theta_r) * math.sin(self.phi_r)
        a_z = math.cos(self.theta_r)

        return np.array([a_x, a_y, a_z]), angle # return axis and angle