import numpy as np
import pandas as pd 

# define dataset here
dataset_filename = "/home/johann/motion-blur-cam-pose-tracker/semester-thesis/RelisticRendering-dataset/groundTruthPoseVel_imu.txt"

# dataset class, responsible for readout ground-truth data at the main locations [0, 0.1, 0.2, ...]
class Dataset():
	def __init__(self):
		super(Dataset, self).__init__()

		self.gt_data = pd.read_csv(dataset_filename, sep ='\s+').values # read from txt-file

		self.interval = 100 # indices between main locations
		self.begin_index = 2 # indices starts at 2

	def get_pose_at(self, timestamp):
		index = int((self.interval) * timestamp * 10 ) # calculate index

		assert (timestamp < 1.4), "no data for this timestamp value"

		#if timestamp >= 1.4:
		#	print("no data for this timestamp value")
		#	return -1

		# extracting quat and tran
		quat = self.gt_data[index, 1:5]
		tran = self.gt_data[index, 5:8]

		return quat, tran