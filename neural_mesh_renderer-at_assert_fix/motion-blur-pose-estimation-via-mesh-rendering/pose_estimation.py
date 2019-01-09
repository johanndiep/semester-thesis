# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This program finds the pose of the blurred input image via an optimization process. It can
# be tested with the additional rendered dataset, for which RGB-, depth- and blurred-images
# are available. Additionally, the informations on camera ground-truth trajectory, extrinsics,
# intrinsics and image logs must be specified in the file 'dataset.py'. In order to simulate 
# practical application, the given depth map can be perturbed by a value within an interval.


# libraries
from lib import dataset
from lib import meshgeneration
from lib import framework
from lib import optimization

# external libraries
import argparse
import torch
import meshio
import math
import numpy as np
from pyquaternion import Quaternion


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', type = int, default  =1)
	args = parser.parse_args()

	print("===========================================================================================")
	print("This program finds the pose of the blurred input image via an optimization process. It can")
	print("be tested with the additional rendered dataset, for which RGB-, depth- and blurred-images")
	print("are available. Additionally, the informations on camera ground-truth trajectory, extrinsics,")
	print("intrinsics and image logs must be specified in the file 'dataset.py'. In order to simulate")
	print("practical application, the given depth map can be perturbed by a value within an interval.")
	print("===========================================================================================")

	# set default tensor
	if args.gpu > 0:
		torch.set_default_tensor_type(torch.cuda.FloatTensor)
		print("*** GPU setting:", bool(args.gpu))

#######################################################################################################

	# hyperparameters to be defined
	cam_index = 0 # cam index [0, 1]
	img_ref = 1 # reference image [1, ..., 13]
	img_cur = 2 # current image [1, ..., 13]
	dist_tran_norm = 0.2 # pertube the initial guess for translation
	dist_angl_norm = 0 # pertube the initial guess for rotation
	scale = 3 # scaling factor for downsizing according to runtime-precision tradeoff [0, ...]
	N_poses = 5 # number of reprojection poses during blurring
	depth_disturbance = 1 # perturb depth by a random value between [-depth_disturbance, depth_disturbance] [m]

	print("*** Following hyperparameters were chosen:")
	print("*** - Camera:", cam_index)
	print("*** - Reference image:", img_ref)
	print("*** - Current image:", img_cur)
	print("*** - Scale:", scale)
	print("*** - Number of blur-poses:", N_poses)
	print("*** - Depth disturbance:", depth_disturbance)

#######################################################################################################

	t_ref = dataset.ImageLogs().get_timestamp(cam_index, img_ref)[0] # reference timestamp
	t_cur = dataset.ImageLogs().get_timestamp(cam_index, img_cur)[0] # current timestamp 

	# choose pose to be estimated
	cur_quat, cur_tran_SE3 = dataset.GroundTruth().get_pose_at(t_cur)
	print("*** Ground-truth pose at current location:")
	print("*** - Translation (SE3 [x, y, z])", cur_tran_SE3)
	print("*** - Rotation (Quaternion [qw, qx, qy, qz]):", cur_quat)
	print("*** Ground-Truth pose will be perturbed by {} for translation and {} for rotation.".format(dist_tran_norm, dist_angl_norm))

	# generate 3D pointcloud and polygon mesh
	print("*** Generating 3D pointcloud and polygon-mesh at scale {}. This might take a while.".format(scale))
	mesh_obj = meshgeneration.MeshGeneration(cam_index, img_ref, t_ref, scale, depth_disturbance)
	pointcloud_ray, faces = mesh_obj.generate_mean_mesh()
	
	# saving 3D pointcloud and polygon mesh
	#np.savetxt('pointcloud.txt', pointcloud_ray)
	#meshio.write_points_cells("polygon_mesh.off", pointcloud_ray, {"triangle": faces})
	#print("*** Saved as 'pointcloud.txt' and 'polygon_mesh.off' for Meshlab visualization.")

	framework_obj = framework.Framework(cam_index, img_ref, img_cur, t_ref, t_cur, pointcloud_ray, faces, dist_tran_norm, dist_angl_norm, cur_quat, cur_tran_SE3, N_poses) # setting up the optimization framework

	# setting up the optimization process
	optimization_obj = optimization.Optimization(framework_obj, cur_tran_SE3, cur_quat)
	solved_tran_SE3, solved_quat = optimization_obj.Adam()

	# results
	print("*** Solved pose:")
	print("*** - Translation (SE3 [x, y, z])", solved_tran_SE3)
	print("*** - Rotation (Quaternion [qw, qx, qy, qz]):", [round(solved_quat[0], 6), round(solved_quat[1], 6), round(solved_quat[2], 6), round(solved_quat[3], 6)])
	print("*** - Translation error [m]:", math.sqrt((solved_tran_SE3[0] - cur_tran_SE3[0]) ** 2 + (solved_tran_SE3[1] - cur_tran_SE3[1]) ** 2 + (solved_tran_SE3[2] - cur_tran_SE3[2]) ** 2))
	print("*** - Rotation error [rad]:", math.fabs(solved_quat.angle - Quaternion(cur_quat).angle))


if __name__ == '__main__':
	main()