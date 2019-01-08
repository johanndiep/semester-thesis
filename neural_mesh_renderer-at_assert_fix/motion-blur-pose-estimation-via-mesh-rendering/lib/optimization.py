# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file is responsible for the optimization algorithm.


# external libraries
import torch
import math
from tqdm import tqdm
from pyquaternion import Quaternion


class Optimization():
	def __init__(self, framework, cur_tran_SE3, cur_quat):
		super(Optimization, self).__init__()

		self.framework = framework # initializing framework
		
		# ground-truth pose
		self.cur_tran_SE3  = cur_tran_SE3
		self.cur_quat = cur_quat

		self.rounds = 1000 # max rounds for iterations

	def Adam(self):
		lr = 0.01 # learning rate
		
		print("*** Start optimization with Adam algorithm with learning rate {}.".format(lr)) # print statement

		optimizer = torch.optim.Adam(self.framework.parameters(), lr = lr) # optimizer to be tuned

		loop = tqdm(range(self.rounds)) # rounds visualization

    	# loop optimization
		for i in loop:

			solved_pose_se3 = self.framework.cur_pose_se3		
			solved_tran_SE3 = (self.framework.se3_exp(solved_pose_se3)[0,:,3]).detach().cpu().numpy()
			solved_tran_error = math.sqrt((solved_tran_SE3[0] - self.cur_tran_SE3[0]) ** 2 + (solved_tran_SE3[1] - self.cur_tran_SE3[1]) ** 2 + (solved_tran_SE3[2] - self.cur_tran_SE3[2]) ** 2)

			solved_aa = (self.framework.cur_pose_se3[3:]).detach().cpu().numpy()
			solved_aa_angle = math.sqrt(solved_aa[0] ** 2 + solved_aa[1] ** 2 + solved_aa[2] ** 2)
			solved_rot_error = math.fabs(solved_aa_angle - Quaternion(self.cur_quat).angle)

			if (solved_tran_error < 0.01 and solved_rot_error < 0.005):
				loop.close()
				break

			optimizer.zero_grad() # set gradients to zero 
        
			# calculating loss function and backpropagating
			loss = self.framework.forward()
			loss.backward()
			optimizer.step()

			loop.set_description("*** Optimizing, current loss at %.4f." % loss.data) # loss print

		solved_quat = Quaternion(axis = solved_aa / solved_aa_angle, angle = solved_aa_angle) 

		return solved_tran_SE3, solved_quat # return solved pose