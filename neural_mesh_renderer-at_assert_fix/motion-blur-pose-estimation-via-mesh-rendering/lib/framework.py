# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file defines the model framework for the optimization process.


# libraries
from lib import randomizer
from lib import dataset
from lib import imagegeneration

# external libraries
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from pyquaternion import Quaternion


store_path = "/home/johann/motion-blur-cam-pose-tracker/semester-thesis/neural_mesh_renderer-at_assert_fix/motion-blur-pose-estimation-via-mesh-rendering/data/"

torch.set_default_tensor_type(torch.cuda.FloatTensor) # using CUDA


# for optimization
class Framework(imagegeneration.ImageGeneration, nn.Module):
	def __init__(self, cam_index, img_ref, img_cur, t_ref, t_cur, pointcloud_ray, faces, dist_tran_norm, dist_angl_norm, cur_quat, cur_tran_SE3, N_poses):
		super(Framework, self).__init__()

		# variables
		self.cam_index = cam_index
		self.img_ref = img_ref
		self.img_cur = img_cur
		self.t_ref = t_ref
		self.t_cur = t_cur
		self.pointcloud_ray = pointcloud_ray
		self.faces = faces
		self.dist_tran_norm = dist_tran_norm
		self.dist_angl_norm = dist_angl_norm
		self.cur_quat = cur_quat
		self.cur_tran_SE3 = cur_tran_SE3
		self.N_poses = N_poses
 
		# get disturbance
		# randomizer_obj = randomizer.Randomizer()
		# dist_tran_SE3 = randomizer_obj.rand_position_offset(self.dist_tran_norm)
		# dist_rot_a, dist_rot_angl = randomizer_obj.rand_rotation_offset(self.dist_angl_norm)
		dist_direction_tran = 1 # [x = 1, -x = 2, y = 3, -y = 4, z = 5, -z = 6]
		dist_direction_rot = 1 # [x = 1, -x = 2, y = 3, -y = 4, z = 5, -z = 6]
		directed_obj = randomizer.Direction(dist_direction_tran, dist_direction_rot)
		dist_tran_SE3 = directed_obj.directed_position_offset(self.dist_tran_norm)
		dist_rot_a, dist_rot_angl = directed_obj.directed_rotation_offset(self.dist_angl_norm)

        # reading in translation in SE(3) form, adding shift and transforming it to se(3) form
		cur_quat = Quaternion(self.cur_quat)
		cur_tran_SE3 = self.cur_tran_SE3 + dist_tran_SE3
		cur_tran_se3 = self.from_SE3t_to_se3u(cur_quat, torch.tensor(cur_tran_SE3).float())

		# reading quaternion, adding shift and transforming it to angle axis form
		dist_quat = Quaternion(axis = dist_rot_a, angle = dist_rot_angl)
		cur_quat = cur_quat * dist_quat
		cur_aa = torch.tensor(cur_quat.axis * cur_quat.angle).float()

		# print disturbed current pose
		print("*** Initial-guess pose at location:")
		print("*** - Translation (SE3 [x, y, z]):", cur_tran_SE3)
		print("*** - Rotation (Quaternion [qw, qx, qy, qz]):", [round(cur_quat[0], 6), round(cur_quat[1], 6), round(cur_quat[2], 6), round(cur_quat[3], 6)])

		# concatenating rotation and translation to tangent form, initializing parameter variable
		self.cur_pose_se3 = nn.Parameter(torch.cat([cur_tran_se3, cur_aa], dim = 0), requires_grad = True)
		
		# reading in blurry image and converting it to torch tensor
		blur_obj = dataset.Blur()
		blur_ref = blur_obj.get_blur_image(self.cam_index, self.img_cur)
		self.blur_ref = torch.tensor(blur_ref).float()

		self.img_saving = 0 # parameter for image saving

	def forward(self):
		blur_image = self.blurrer(self.cam_index, self.img_ref, self.img_cur, self.t_ref, self.t_cur, self.pointcloud_ray, self.faces, self.cur_pose_se3, self.N_poses) # calling the blurrer

		# displaying the generated blurred image
		plot_blur_image = blur_image
		# cv2.imshow('image', plot_blur_image.detach().cpu().numpy().astype(np.uint8))
		# cv2.waitKey(10000)
		# cv2.destroyAllWindows()

		# store intermediate results
		store_path_filename = os.path.join(store_path, "artificial_blur_%d.jpg"%self.img_saving)
		cv2.imwrite(store_path_filename, plot_blur_image.detach().cpu().numpy().astype(np.uint8))
		self.img_saving = self.img_saving + 1

		loss = torch.sum((blur_image - self.blur_ref) * (blur_image - self.blur_ref)) # loss function, sum of quadratic deviation

		return loss


# just for generating images
class Framework_image_generator(imagegeneration.ImageGeneration):
	def __init__(self, cam_index, img_ref, img_cur, t_ref, t_cur, pointcloud_ray, faces, cur_quat, cur_tran_SE3, sharp, N_poses):
		super(Framework_image_generator, self).__init__()

		# variables
		self.cam_index = cam_index
		self.img_ref = img_ref
		self.img_cur = img_cur
		self.t_ref = t_ref
		self.t_cur = t_cur
		self.pointcloud_ray = pointcloud_ray
		self.faces = faces
		self.cur_quat = cur_quat
		self.cur_tran_SE3 = cur_tran_SE3
		self.N_poses = N_poses

		# reading in translation in SE(3) form and transforming it to se(3) form
		cur_quat = Quaternion(self.cur_quat)
		cur_tran_SE3 = self.cur_tran_SE3
		cur_tran_se3 = self.from_SE3t_to_se3u(cur_quat, torch.tensor(cur_tran_SE3).float())

		# reading quaternion and transforming it to angle axis form
		cur_aa = torch.tensor(cur_quat.axis * cur_quat.angle).float()

		# print statement
		print("Generate an image at location:")
		print("*** - Translation (SE3 [x, y, z]):", cur_tran_SE3)
		print("*** - Rotation (Quaternion [qw, qx, qy, qz]):", [round(cur_quat[0], 6), round(cur_quat[1], 6), round(cur_quat[2], 6), round(cur_quat[3], 6)])

		# concatenating rotation and translation to tangent form, initializing parameter variable
		self.cur_pose_se3 = torch.cat([cur_tran_se3, cur_aa], dim = 0)

		depth = self.render_depth(self.cam_index, self.pointcloud_ray, self.faces, self.cur_pose_se3) # generating depth image at that pose

		if sharp == True:
			image = self.reprojector(self.cam_index, self.img_ref, self.t_ref, depth, self.cur_pose_se3) # generate image at that pose
		else:
			image = self.blurrer(self.cam_index, self.img_ref, self.img_cur, self.t_ref, self.t_cur, self.pointcloud_ray, self.faces, self.cur_pose_se3, self.N_poses) # calling the blurrer

		# displaying
		cv2.imshow('image', image.detach().cpu().numpy().astype(np.uint8))
		cv2.waitKey(10000)
		cv2.destroyAllWindows()