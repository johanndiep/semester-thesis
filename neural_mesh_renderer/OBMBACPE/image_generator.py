# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This program generates (sharp/blurry) images at arbitrary poses. It requires a reference and
# its corresponding depth image. Additionally, the information on camera ground-truth at the 
# reference position as well as the camera extrinsics and intrinsics must be availabe. In order 
# to simulate practical application, the given depth map can be perturbed by a value within an 
# interval.


# libraries
from lib import dataset
from lib import meshgeneration
from lib import framework

# external libraries
import argparse
import torch
import meshio
import time
import numpy as np


def main():
	#parsing
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', type = int, default  = 1)
	args = parser.parse_args()

	# print statements
	print("===========================================================================================")
	print("This program generates (sharp/blurry) images at arbitrary poses. It requires a reference and")
	print("its corresponding depth image. Additionally, the information on camera ground-truth at the")
	print("reference position as well as the camera extrinsics and intrinsics must be availabe. In order")
	print("to simulate practical application, the given depth map can be perturbed by a value within an")
	print("interval.")
	print("===========================================================================================")

	# set default tensor
	if args.gpu > 0:
		torch.set_default_tensor_type(torch.cuda.FloatTensor)
		print("*** GPU setting:", torch.cuda.is_available())

#######################################################################################################

	# hyperparameters to be defined
	cam_index = 0 # cam index [0, 1]
	img_ref = 1 # reference image [1, ..., 13]
	img_cur = 2 # current image [1, ..., 13]
	scale = 3 #scaling factor for downsizing according to runtime-precision tradeoff [0, ...]
	N_poses = 5 # number of reprojection poses during blurring
	depth_variance = 0 # perturb depth by a random value between [-depth_variance, depth_variance] [m]
	sharp = False # if set to false, a blurry image is generated
	pyramid_scale = 0 # apply coarse to sparse solving principle

	# print statements
	print("*** Following hyperparameters were chosen:")
	print("*** - Camera:", cam_index)
	print("*** - Reference image:", img_ref)
	print("*** - Current image:", img_cur)
	print("*** - Scale for mesh construction:", scale)
	print("*** - Depth variance:", depth_variance)
	print("*** - Pyramid scale:", pyramid_scale)
	if sharp == True:
		print("*** - Image type: sharp")
	else:
		print("*** - Image type: blurry")

#######################################################################################################

	t_ref = dataset.ImageLogs().get_timestamp(cam_index, img_ref)[0] # reference timestamp
	t_cur = dataset.ImageLogs().get_timestamp(cam_index, img_cur)[0] # current timestamp

	# choose pose where an image should be generated
	cur_quat, cur_tran_SE3 = dataset.GroundTruth().get_pose_at(t_cur)

	# generate 3D pointcloud and polygon mesh
	print("*** Generating 3D pointcloud and polygon-mesh at scale {}. This might take a while.".format(scale))
	start_time = time.time()
	mesh_obj = meshgeneration.MeshGeneration(cam_index, img_ref, t_ref, scale, depth_variance)
	pointcloud_ray, faces = mesh_obj.generate_mean_mesh()
	end_time = time.time()
	print("*** Generating 3D representation took {} seconds to complete.".format(end_time - start_time))

	# saving 3D pointcloud and polygon mesh
	np.savetxt('pointcloud.txt', pointcloud_ray)
	meshio.write_points_cells("polygon_mesh.off", pointcloud_ray, {"triangle": faces})
	print("*** Saved as 'pointcloud.txt' and 'polygon_mesh.off' for Meshlab visualization.")

	# setting up the generator framework
	start_time = time.time()
	framework_ig_obj = framework.Framework_image_generator(cam_index, img_ref, img_cur, t_ref, t_cur, pointcloud_ray, faces, cur_quat, cur_tran_SE3, sharp, N_poses, pyramid_scale)
	end_time = time.time()
	print("*** Generating image took {} seconds to complete.".format(end_time - start_time)) 


if __name__ == '__main__':
	main()