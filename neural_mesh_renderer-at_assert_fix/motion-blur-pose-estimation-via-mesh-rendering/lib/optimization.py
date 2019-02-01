# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file is responsible for the optimization algorithm.


# external libraries
import torch
import math
from tqdm import tqdm
from pyquaternion import Quaternion


torch.set_default_tensor_type(torch.cuda.FloatTensor) # using CUDA


class Optimization():
	def __init__(self, framework, cur_tran_SE3, cur_quat):
		super(Optimization, self).__init__()

		self.framework = framework # initializing framework
		
		# ground-truth pose
		self.cur_tran_SE3  = cur_tran_SE3
		self.cur_quat = cur_quat

		self.rounds = 100 # max rounds for iterations
		self.minimal_loss = 99999999999 # random high number"
		self.convergence = False # convergence flag"

	def Adam(self):
		lr = 0.01 # learning rate
		
		print("*** Start optimization with Adam algorithm with learning rate {}.".format(lr)) # print statement

		optimizer = torch.optim.Adam(self.framework.parameters(), lr = lr) # optimizer to be tuned

		loop = tqdm(range(self.rounds)) # rounds visualization

    	# loop optimization
		for i in loop:
			# current loop translation
			solved_pose_se3 = self.framework.cur_pose_se3
			solved_tran_SE3 = (self.framework.se3_exp(solved_pose_se3)[0,:,3]).detach().cpu().numpy()
			
			solved_tran_error = math.sqrt((solved_tran_SE3[0] - self.cur_tran_SE3[0]) ** 2 + (solved_tran_SE3[1] - self.cur_tran_SE3[1]) ** 2 + (solved_tran_SE3[2] - self.cur_tran_SE3[2]) ** 2) # current loop translation error

			minimal_tran_SE3 = solved_tran_SE3 # copy
			
			# current loop rotation
			solved_aa = (self.framework.cur_pose_se3[3:]).detach().cpu().numpy()
			solved_aa_angle = math.sqrt(solved_aa[0] ** 2 + solved_aa[1] ** 2 + solved_aa[2] ** 2)

			solved_rot_error = math.fabs(solved_aa_angle - Quaternion(self.cur_quat).angle) # current loop rotation error

			minimal_aa = solved_aa # copy
			minimal_aa_angle = solved_aa_angle # copy

			# convergence condition
			if (solved_tran_error < 0.01 and solved_rot_error < 0.02):
				loop.close()
				self.convergence = True
				break

			optimizer.zero_grad() # set gradients to zero 
        
			# calculating loss function and backpropagating
			loss = self.framework.forward()
			loss.backward()
			optimizer.step()

			# if no convergence, take the closest solution
			if loss.data < self.minimal_loss:
				self.minimal_loss = loss.data # replace minimal loss so far

				# save pose
				minimal_tran_SE3 = solved_tran_SE3
				minimal_aa  = solved_aa
				minimal_aa_angle = solved_aa_angle
				minimal_pose_se3 = self.framework.cur_pose_se3


			loop.set_description("*** Optimizing, current loss at %.4f." % loss.data) # loss print

		# if convergence
		if self.convergence == True:
			print(solved_pose_se3)
			solved_quat = Quaternion(axis = solved_aa / solved_aa_angle, angle = solved_aa_angle) 
			return solved_tran_SE3, solved_quat # return solved pose

		# if no convergence
		else:
			print(minimal_pose_se3)
			minimal_quat = Quaternion(axis = minimal_aa / minimal_aa_angle, angle = minimal_aa_angle)
			return minimal_tran_SE3, minimal_quat