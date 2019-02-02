# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file is responsible for generating a 3D polygon mesh and depth images for specific poses.
# The mesh size can be downsized by a power of 2 in order to generate less faces for 
# computational efficiency.


# libraries
from lib import dataset

# external libraries
import torch
import meshzoo
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from skimage.transform import resize


torch.set_default_tensor_type(torch.cuda.FloatTensor) # using CUDA


# generating a 3D mesh and depth images from specific poses, downsize by power of 2 in order to generate 
# less faces, which results in faster computation
class MeshGeneration(dataset.Intrinsics, dataset.GroundTruth, dataset.Extrinsics, dataset.Perturb):
    def __init__(self, cam_index, img_ref, t_ref, scale, depth_variance):
        super(MeshGeneration, self).__init__()

        # reading input
        self.cam_index = cam_index
        self.img_ref = img_ref
        self.t_ref = t_ref
        self.scale = scale
        self.depth_variance = depth_variance

        self.scale_power = 2 ** self.scale # downsizing by power of 2

        # calibration matrix
        img_size_x, img_size_y, K = self.get_intrinsics(cam_index)
        K = torch.tensor(K).float()

        self.K_scaled = K / self.scale_power # downsize K

        # downsize image size
        self.img_size_x_scaled = int(img_size_x // self.scale_power)
        self.img_size_y_scaled = int(img_size_y // self.scale_power)

    # generate the polygon-mesh
    def generate_mean_mesh(self):
        _, faces = meshzoo.rectangle(xmin = -1, xmax = 1, ymin = -1, ymax = 1., nx = self.img_size_x_scaled, ny = self.img_size_y_scaled, zigzag = True) # generate a mesh connecting each neighbooring pixel in a rectangular way

        # torch tensor of dimension 1  
        x = torch.arange(0, self.img_size_x_scaled, 1).float()
        y = torch.arange(0, self.img_size_y_scaled, 1).float()

        # precalculation for 3D projection
        x_ = (x - self.K_scaled[0][2]) / self.K_scaled[0][0]
        y_ = (y - self.K_scaled[1][2]) / self.K_scaled[1][1]

        # torch tensor of dimension 2
        xx = x_.repeat(self.img_size_y_scaled, 1)
        yy = y_.view(self.img_size_y_scaled, 1).repeat(1, self.img_size_x_scaled)
        zz = torch.ones_like(xx)

        # calculating the absolute position in 3D space in camera-frame
        xx, yy, zz = self.absolute_mesh(xx, yy, zz)

        # concatenating the 3D points 
        pointcloud_ray = torch.stack([xx, yy, zz], dim=-1)
        pointcloud_ray = pointcloud_ray.view(-1, 3)

        # reading start_quat and cam_quat
        ref_quat, ref_tran_SE3 = self.get_pose_at(self.t_ref)
        cam_quat, cam_tran_SE3 = self.get_extrinsics(self.cam_index)[:4]

        # reading start- and camera-orientation
        ref_quat = Quaternion(ref_quat)
        cam_quat = Quaternion(cam_quat)

        # transforming 3D points form camera- to world-frame
        for index in range(0, pointcloud_ray.shape[0]):
            rotated_point = ref_quat.rotate(cam_quat.rotate(pointcloud_ray[index]) + cam_tran_SE3)
            pointcloud_ray[index] = torch.tensor(rotated_point).float() + torch.tensor(ref_tran_SE3).float()

        return pointcloud_ray, torch.tensor(faces).int() # return pointcloud and faces

    # generating the absolute distances
    def absolute_mesh(self, xx, yy, zz):

        # reading depth, resizing it and storing it in a dataframe
        depth_values = self.get_perturb_depth(self.cam_index, self.img_ref, self.depth_variance)
        depth_values = resize(depth_values, (self.img_size_y_scaled, self.img_size_x_scaled), mode = 'constant')

        depth_tensor = torch.tensor(depth_values).float() # making it a torch tensor

        # display depth image
        # test = depth_tensor
        # test = test.detach().cpu().numpy()
        # plt.imshow(test)
        # plt.show()

        # 3D projection
        zz = depth_tensor / torch.sqrt(xx * xx + yy * yy + 1)
        xx = zz * xx
        yy = zz * yy

        return xx, yy, zz # return 3D points in camera-frame