# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file is responsible for the optimization algorithm.


# external libraries
import torch
import math
from tqdm import tqdm
from pyquaternion import Quaternion

class Optimization():
	def __init__(self, framework):
		super(Optimization, self).__init__()

		self.framework = framework # initializing framework
		self.rounds = 1000 # max rounds for iterations

	def Adam(self):
		lr = 0.01 # learning rate
		min_loss = 9999999999 # random high number
		break_iterator = 0 # iteration variable for break
		break_condition = 50 # break if loss does not get lower for number of iterations 
		
		print("*** Start optimization with Adam algorithm with learning rate {}.".format(lr)) # print statement

		optimizer = torch.optim.Adam(self.framework.parameters(), lr = lr) # optimizer to be tuned

		loop = tqdm(range(self.rounds)) # rounds visualization

    	# loop optimization
		for i in loop:
			optimizer.zero_grad() # set gradients to zero 
        
			# calculating loss function and backpropagating
			loss = self.framework.forward()
			loss.backward()
			optimizer.step()

			loop.set_description("*** Optimizing, current loss at %.4f." % loss.data) # loss print

			# breaking condition
			if loss.data < min_loss:
				min_loss = loss.data
				break_iterator = 0
				min_pose_se3 = self.framework.cur_pose_se3
			else:
				break_iterator = break_iterator + 1
			
				# break out of the loop
				if break_iterator > break_condition:
					loop.close()
					break

		# solved pose
		min_tran_SE3 = self.framework.se3_exp(min_pose_se3)[0,:,3]
		min_aa = (min_pose_se3[3:]).detach().cpu().numpy()
		min_aa_angle = math.sqrt(min_aa[0] ** 2 + min_aa[1] ** 2 + min_aa[2] ** 2)

		min_quat = Quaternion(axis = min_aa / min_aa_angle, angle = min_aa_angle) 

		return (min_tran_SE3).detach().cpu().numpy(), min_quat # return solved pose