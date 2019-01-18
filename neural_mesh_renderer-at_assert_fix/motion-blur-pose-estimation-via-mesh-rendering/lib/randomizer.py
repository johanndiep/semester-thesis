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

    # random translation vector
    def rand_position_offset(self, dist): 
        # calculating cartesian coordinates from spherical coordinates 
        x = dist * math.sin(self.theta_t) * math.cos(self.phi_t)
        y = dist * math.sin(self.theta_t) * math.sin(self.phi_t)
        z = dist * math.cos(self.theta_t)

        return np.array([x, y, z]) # return cartesian coordinates

    # random angle axis vector
    def rand_rotation_offset(self, angle):
        # calculating cartesian coordinates from spherical coordinates 
        a_x = math.sin(self.theta_r) * math.cos(self.phi_r)
        a_y = math.sin(self.theta_r) * math.sin(self.phi_r)
        a_z = math.cos(self.theta_r)

        return np.array([a_x, a_y, a_z]), angle # return axis and angle


# calculates cardinal direction with desired length
class Direction():
	def __init__(self, direction_tran, direction_rot):
		# saves desired direction
		self.direction_tran = direction_tran # [x = 1, -x = 2, y = 3, -y = 4, z = 5, -z = 6]
		self.direction_rot = direction_rot # [x = 1, -x = 2, y = 3, -y = 4, z = 5, -z = 6]

	# directed translation vector
	def directed_position_offset(self, dist):
		# calculating cartesian coordinates
		if self.direction_tran == 1:
			return dist * np.array([1, 0, 0])
		if self.direction_tran == 2:
			return dist * np.array([-1, 0, 0])
		if self.direction_tran == 3:
			return dist * np.array([0, 1, 0])
		if self.direction_tran == 4:
			return dist * np.array([0, -1, 0])
		if self.direction_tran == 5:
			return dist * np.array([0, 0, 1])
		if self.direction_tran == 6:
			return dist * np.array([0, 0, -1])

	# directed angle axis vector
	def directed_rotation_offset(self, angle):
		# calculating cartesian coordinates
		if self.direction_rot == 1:
			return np.array([1, 0, 0]), angle
		if self.direction_rot == 2:
			return np.array([-1, 0, 0]), angle
		if self.direction_rot == 3:
			return np.array([0, 1, 0]), angle
		if self.direction_rot == 4:
			return np.array([0, -1, 0]), angle
		if self.direction_rot == 5:
			return np.array([0, 0, 1]), angle
		if self.direction_rot == 6:
			return np.array([0, 0, -1]), angle