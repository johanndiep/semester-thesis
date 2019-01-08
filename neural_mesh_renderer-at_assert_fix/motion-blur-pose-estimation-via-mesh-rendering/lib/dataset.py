# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file is responsible for reading out:
# * ground-truth trajectory information
# * extrinsics of the camera: transformation from camera to vehicle body frame 
# * intrinsics of the camera: resolution, calibration matrix
# * image-logs: timestamp and exposure time for each image of each camera
# * ground-truth depth images


# external libraries
import os
import numpy as np
import pandas as pd
import cv2


# define dataset here
dataset_filename = "/home/johann/motion-blur-cam-pose-tracker/semester-thesis/RelisticRendering-dataset/"


# class responsible for readout ground-truth data at the main locations [0, 0.1, 0.2, ...]
class GroundTruth():
	def __init__(self):
		super(GroundTruth, self).__init__()

		# path to trajectory file
		gt_filename = os.path.join(dataset_filename, "groundTruthPoseVel_imu.txt")

		self.gt_data = pd.read_csv(gt_filename, sep ='\s+').values # read from txt-file

		self.interval = 100 # indices between main locations
		self.begin_index = 2 # indices starts at 2

	def get_pose_at(self, timestamp):
		index = int((self.interval) * timestamp * 10) # calculate index

		assert (timestamp < 1.4), "No data for this timestamp value." # error message for wrong value

		# extracting quat and tran
		quat = self.gt_data[index, 1:5]
		tran = self.gt_data[index, 5:8]

		return quat, tran # return quat and tran


# define all the extrinsic parameters of the camera
class Extrinsics():
	def __init__(self):
		super(Extrinsics, self).__init__()

		self.scalingFactor = 0.01 # transform [cm] to [m]

		# [qw qx qy qz x y z (cm)] from camera frame to vehicle body frame
		self.camera_extrinsics_0 = [0.5, -0.5, 0.5, -0.5, 0 * self.scalingFactor, 0 * self.scalingFactor, 0 * self.scalingFactor]
		self.camera_extrinsics_1 = [0.5, -0.5, 0.5, -0.5, 0 * self.scalingFactor, -40 * self.scalingFactor, 0 * self.scalingFactor]

	# return value depending on camera
	def get_extrinsics(self, cam_index):
		if cam_index == 0:
			return self.camera_extrinsics_0[:4], self.camera_extrinsics_0[4:]
		else:
			return self.camera_extrinsics_1[:4], self.camera_extrinsics_1[4:]


# define all the intrinsic parameters of the camera
class Intrinsics():
	def __init__(self):
		super(Intrinsics, self).__init__()

		# resolution of the images, x-cols and y-rows
		self.img_size_x_0 = 640
		self.img_size_x_1 = 640
		self.img_size_y_0 = 480
		self.img_size_y_1 = 480

		# calibration matrix according to pinhole camera model
		self.calibration_matrix_0 = [[320., 0, 320.], [0., 320., 240.], [0., 0., 1.]]
		self.calibration_matrix_1 = [[320., 0, 320.], [0., 320., 240.], [0., 0., 1.]]

	# return values depending on camera
	def get_intrinsics(self, cam_index):
		if cam_index == 0:
			return self.img_size_x_0, self.img_size_y_0, self.calibration_matrix_0
		if cam_index == 1:
			return self.img_size_x_1, self.img_size_y_1, self.calibration_matrix_1


# contains the log informations of the images
class ImageLogs():
	def __init__(self):
		super(ImageLogs, self).__init__()

		self.exposure_time = 0.04 # assumed to be equal for all blur images
		self.first_timestamp = 0.1 # timestamp for first image

	# asssume both cam has the same timestamp for each image	
	def get_timestamp(self, cam_index, image_index):
		return self.first_timestamp * image_index, self.exposure_time


# reading depth images from dataset
class Depth():
	def __init__(self):
		super(Depth, self).__init__()

		self.d_filename = os.path.join(dataset_filename, "depth") # path to depth folder

	def get_depth_map(self, cam_index, image_index):
		self.depth_filename = os.path.join(self.d_filename, "cam%s"%cam_index,"depth_map_%s.csv"%image_index) # exact depth map location
		
		depth_df = pd.read_csv(self.depth_filename, sep = '\s+', header = None) # reading and storing it in pandas dataframe format
		
		return depth_df.values # returning depth map


# read blur image
class Blur():
	def __init__(self):
		super(Blur, self).__init__()

		self.b_filename = os.path.join(dataset_filename, "blurred") # path to blur folder

	def get_blur_image(self, cam_index, image_index):

		self.blur_filename = os.path.join(self.b_filename, "cam%s"%cam_index, "%s.png"%image_index) # exact blur image location

		blur_image = cv2.imread(self.blur_filename, 0) # read blur image

		return blur_image # return blur image


# read rgb image
class Sharp():
	def __init__(self):
		super(Sharp, self).__init__()

		self.s_filename = os.path.join(dataset_filename, "rgb") # path to rgb folder

	def get_sharp_image(self, cam_index, image_index):

		self.sharp_filename = os.path.join(self.s_filename, "cam%s"%cam_index, "%s.png"%image_index) # exact rgb image location

		sharp_image = cv2.imread(self.sharp_filename, 0) # read rgb image and convert to grayscale

		return sharp_image # return grayscale image


# perturb depth map
class Perturb(Depth):
	def __init__(self):
		super(Perturb, self).__init__()

	def perturb_depth(self, cam_index, image_index, depth_disturbance):

		depth = self.get_depth_map(cam_index, image_index)

		disturbance_matrix = np.random.uniform(low = -depth_disturbance, high = depth_disturbance, size = depth.shape)

		depth_perturbed = depth + disturbance_matrix

		return depth_perturbed