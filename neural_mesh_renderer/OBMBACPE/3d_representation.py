# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This program produces a 3D pointcloud and polygon-mesh representation of the environment in the
# intial coordinate-frame. The generated txt- and obj-file can be observed in Meshlab. The requirement
# is the availablity of a sharp reference image with its corresponding depth-map. Additionally, 
# the information on camera ground-truth at the reference position as well as the camera extrinsics
# and intrinsics must be availabe. In order to simulate practical application, the given depth 
# map can be perturbed by a value within an interval.


# libraries
from lib import dataset
from lib import meshgeneration

# external libraries
import argparse
import torch
import meshio
import numpy as np
import time

def main():
	#parsing
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', type = int, default  = 1)
	args = parser.parse_args()

	# print statements
	print("===========================================================================================")
	print("This program produces a 3D pointcloud and polygon-mesh representation of the environment in the")
	print("intial coordinate-frame. The generated txt- and obj-file can be observed in Meshlab. The requirement")
	print("is the availablity of a sharp reference image with its corresponding depth-map. Additionally,")
	print("the information on camera ground-truth at the reference position as well as the camera extrinsics")
	print("and intrinsics must be availabe. In order to simulate practical application, the given depth")
	print("map can be perturbed by a value within an interval.")
	print("===========================================================================================")

	# set default tensor
	if args.gpu > 0:
		torch.set_default_tensor_type(torch.cuda.FloatTensor)
		print("*** GPU setting:", torch.cuda.is_available())

#######################################################################################################

	# hyperparameters to be defined
	cam_index = 0 # cam index [0, 1]
	img_ref = 1 # reference image [1, ..., 13]
	scale = 3 # scaling factor for downsizing according to runtime-precision tradeoff [0, ...]
	depth_variance = 0 # perturb depth by a random value between [-depth_variance, depth_variance] [m]

	# print statements
	print("*** Following hyperparameters were chosen:")
	print("*** - Camera:", cam_index)
	print("*** - Reference image:", img_ref)
	print("*** - Scale for mesh construction:", scale)
	print("*** - Depth variance:", depth_variance)

#######################################################################################################

	t_ref = dataset.ImageLogs().get_timestamp(cam_index, img_ref)[0] # reference timestamp

	# generate 3D pointcloud and polygon mesh
	print("*** Generating 3D pointcloud and polygon-mesh at scale {}. This might take a while.".format(scale))
	start_time = time.time()
	mesh_obj = meshgeneration.MeshGeneration(cam_index, img_ref, t_ref, scale, depth_variance)
	pointcloud_ray, faces = mesh_obj.generate_mean_mesh()
	end_time = time.time()

	# saving 3D pointcloud and polygon mesh
	np.savetxt('pointcloud.txt', pointcloud_ray)
	meshio.write_points_cells("polygon_mesh.off", pointcloud_ray, {"triangle": faces})
	print("*** Saved as 'pointcloud.txt' and 'polygon_mesh.off' for Meshlab visualization.")
	print("*** This process took {} seconds to complete.".format(end_time - start_time))


if __name__ == '__main__':
	main()