#Semester Thesis: Implementation of the Motion-blur Aware Camera-Pose Estimation Algorithm in Python via Mesh Rendering
#Author: Johann Diep (jdiep@student.ethz.ch)

# listing all necessary libraries
import argparse #
import torch #
import torch.nn as nn #
import os # 
import meshzoo # 
import meshio #
import numpy as np # 
import sys
import neural_renderer as nr
import pandas as pd # 
import matplotlib.pyplot as plt
import tqdm
import cv2
import math #
import random # 
from skimage.io import imsave, imread
from skimage.viewer import ImageViewer
from scipy.misc import imshow
from pyquaternion import Quaternion #
from skimage.transform import resize # 
from tqdm import tqdm
from torch.autograd import Variable


# listing all paths for data retrieving and storage
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

# TODO: easier access
depth_dir = '/home/johann/motion-blur-cam-pose-tracker/semester-thesis/RelisticRendering-dataset/depth/cam0/depth_map_1.csv' # used for 3D mesh generation
img_dir = '/home/johann/motion-blur-cam-pose-tracker/semester-thesis/RelisticRendering-dataset/rgb/cam0/1.png' # used for image warping
blur_dir = '/home/johann/motion-blur-cam-pose-tracker/semester-thesis/RelisticRendering-dataset/blurred/cam0/2.png' # used as reference blurry image

# storing the camera parameters of the Realistic Rendering dataset
class CameraParameter():
    
    def __init__(self):
        super(CameraParameter, self).__init__() 

        self.K = torch.tensor([[320., 0, 320.], [0., 320., 240.], [0., 0., 1.]]).float().cuda() # calibration matrix

        # image scale
        self.img_size_x = 640
        self.img_size_y = 480

        self.scale = 3 # scaling factor for downsizing set according to runtime-precision tradeoff

        self.dist_coeffs = torch.tensor([0, 0, 0, 0, 0]).float().cuda() # distortion coefficients, not relevant

        # reference pose extracted into quaternion and translation part, set for your application
        self.start_pose = np.array([0.93579, 0.0281295, 0.0740478, -0.343544, -0.684809, 1.59021, 0.91045])
        self.start_quat = Quaternion(self.start_pose[:4])
        self.start_tran = self.start_pose[4:]

        # current pose information, set for your application
        self.cur_timestamp = 0.2
        self.start_timestamp = 0.1

        # texture parameter, not relevant
        self.texture_size = 2
        
        # camera pose of cam0 extracted into quaternion and translation part, from extrinsics
        # TODO: include cam1, translation part is in [cm]
        cam_pose = np.array([0.5, -0.5, 0.5, -0.5, 0, 0, 0])
        self.cam_quat = Quaternion(cam_pose[:4])
        self.cam_tran = cam_pose[4:]

        self.N_poses = 5 # number of reprojection poses during blurring, set for your application
        self.t_exp = 0.04 # exposure time
        self.t_int = self.cur_timestamp - self.start_timestamp # time interval between two consecutive image-frames


# pose calculation from Lie group to homogeneous transformation
class PoseTransformation():

    def __init__(self):
        super(PoseTransformation, self).__init__()

    def se3_exp(self, tangent):
        # set to dimension 2
        if tangent.dim() < 2:
            tangent = tangent.unsqueeze(dim = 0)

        # extract se(3)-translation and se(3)-rotation
        t = tangent[:, :3]
        phi = tangent[:, 3:]

        R, R_jac = self.so3_exp(phi) # calculating SO(3)-rotation matrix and V
        t = t.unsqueeze(dim = 2) # set to the right dimension

        trans = torch.bmm(R_jac, t) # calculating SE(3)-translation
        return torch.cat([R, trans], dim = 2) # return SO(3)-pose

    def so3_exp(self, phi):
        # set to dimension 2
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim = 0)

        # V matrix size
        jac = phi.__class__(phi.shape[0], 3, 3)
        I = torch.eye(3).expand_as(jac)

        # norm of se(3)-rotation
        angle = phi.norm(p = 2, dim = 1)
        angle = angle + 1e-5

        # calculating the taylor expansions
        s = angle.sin()
        c = angle.cos()
        s_div_angle = s / angle
        one_minus_s_div_angle = 1. - s_div_angle
        one_minus_c = 1. - c
        one_minus_c_div_angle = one_minus_c / angle
        
        # set to the right dimension
        s_div_angle = s_div_angle.unsqueeze(1).unsqueeze(2)
        one_minus_s_div_angle = one_minus_s_div_angle.unsqueeze(1).unsqueeze(2)
        one_minus_c_div_angle = one_minus_c_div_angle.unsqueeze(1).unsqueeze(2)
        c = c.unsqueeze(1).unsqueeze(2)
        s = s.unsqueeze(1).unsqueeze(2)
        one_minus_c = one_minus_c.unsqueeze(1).unsqueeze(2)

        # get unit axis of rotation
        angle = angle.unsqueeze(1)
        axis = phi / angle

        axis2 = self.outer(axis, axis) # calculate squared skew-matrix 
        wedge_axis = self.wedge(axis) # calculate skew_matrix

        # calculating the taylor expansions
        A_jac = s_div_angle * I
        B_jac = one_minus_s_div_angle * axis2
        C_jac = one_minus_c_div_angle * wedge_axis
        A = c * I
        B = one_minus_c * axis2
        C = s * wedge_axis

        return A + B + C, A_jac + B_jac + C_jac # return the SO(3)-rotation and V-matrix

    def outer(self, vecs1, vecs2):
        # set to dimension 2
        if vecs1.dim() < 2:
            vecs1 = vecs1.unsqueeze(dim = 0)
        if vecs2.dim() < 2:
            vecs2 = vecs2.unsqueeze(dim = 0)

        # error message if size is wrong
        if vecs1.shape[0] != vecs2.shape[0]:
            raise ValueError("Inconsistent batch sizes {} and {}".format(vecs1.shape[0], vec2.shape[0]))

        # return squared skew-matrix
        return torch.bmm(vecs1.unsqueeze(dim = 2), vecs2.unsqueeze(dim = 2).transpose(2, 1))[0,:,:]

    def wedge(self, phi):
        # set to dimension 2
        if phi.dim() < 2:
            phi.unsqueeze(dim = 0)

        # error message if size is wrong
        if phi.shape[1] != 3:
            raise ValueError("phi must have shape ({},) or (N,{})".format(3, 3))

        # return skew-matrix 
        Phi = phi.__class__(phi.shape[0], 3, 3)
        Phi[0, :, :] = torch.zeros(3, 3)
        Phi[:, 0, 1] = -phi [:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]

        return Phi[0, :, :] # reduce dimension

    def from_SE3t_to_se3u(self, q, t):
        # reading input in the right form
        quat = Quaternion(q[:4])
        t = t.unsqueeze(dim = 0).unsqueeze(dim = 1).cuda().float()

        # norm and rotation-axis of se(3)-rotation 
        axis = torch.tensor(quat.axis).cuda().float().unsqueeze(dim = 0)
        angle = torch.tensor([quat.angle]).cuda().float().unsqueeze(dim = 0)

        # calculate the taylor expansions
        A = angle.sin() / angle
        B = (1 - angle.cos()) / (angle * angle)
        C = 1 / (angle * angle) *  (1 - A / (2 * B))

        # calculating the inverse V
        I = torch.eye(3).cuda().unsqueeze(dim = 0)
        V_inverse = I - 0.5 * self.wedge(axis * angle) + C * torch.bmm(self.wedge(axis * angle).unsqueeze(dim = 0), self.wedge(axis * angle).unsqueeze(dim = 0))

        return torch.bmm(V_inverse, t.transpose(1,2))[0,:,0] # return se(3)-translation

    def from_SE3rot_to_se3w(self, r):
        # reading and storing rotional elements
        m_00 = r[:, 0, 0]
        m_11 = r[:, 1, 1]
        m_22 = r[:, 2, 2]
        m_01 = r[:, 0, 1]
        m_02 = r[:, 0, 2]
        m_10 = r[:, 1, 0]
        m_12 = r[:, 1, 2]
        m_20 = r[:, 2, 0]
        m_21 = r[:, 2, 1]

        # transforming rotational matrix to angle axis form
        angle = ((m_00 + m_11 + m_22 -1) / 2).acos()
        ax = (m_21 - m_12) / ((m_21 - m_12).pow(2) + (m_02 - m_20).pow(2) + (m_10 - m_01).pow(2)).sqrt()
        ay = (m_02 - m_20) / ((m_21 - m_12).pow(2) + (m_02 - m_20).pow(2) + (m_10 - m_01).pow(2)).sqrt() 
        az = (m_10 - m_01) / ((m_21 - m_12).pow(2) + (m_02 - m_20).pow(2) + (m_10 - m_01).pow(2)).sqrt()

        return torch.cat([ax * angle, ay * angle, az * angle], dim = 0).unsqueeze(dim = 0) # return angle axis 


# defining the renderer
class Renderer(CameraParameter, nn.Module, PoseTransformation):
    def __init__(self, vertices, faces):
        super(Renderer, self).__init__()

        # initializing buffers, only vertices and faces are important, dont care about texture
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])
        textures = torch.ones(1, self.faces.shape[1], self.texture_size, self.texture_size, self.texture_size, 3,
                              dtype = torch.float32)
        self.register_buffer('textures', textures)

        # initialzing ProjectiveRenderer-object
        renderer = nr.ProjectiveRenderer(image_size = self.img_size_x, K = torch.unsqueeze(self.K, 0))
        self.renderer = renderer


# generating a 3D mesh and depth images from specific poses, downsize by power of 2 in order to generate less faces, which results in faster computation
# scale can be set in CameraParameter-class
class MeshGeneration(CameraParameter):
    def __init__(self):
        super(MeshGeneration, self).__init__()

        self.scale_power = 2 ** self.scale # downsizing by power of 2
        
        self.K_scaled = self.K / self.scale_power # downsize K

        # downsize image size
        self.img_size_x_scaled = int(self.img_size_x // self.scale_power)
        self.img_size_y_scaled = int(self.img_size_y // self.scale_power)

    def generate_mean_mesh(self):
        _, faces = meshzoo.rectangle(xmin = -1, xmax = 1,
                                     ymin = -1, ymax = 1.,
                                     nx = self.img_size_x_scaled, ny = self.img_size_y_scaled,
                                     zigzag = True) # generate a mesh connecting each neighbooring pixel in a rectangular way

        # torch tensor of dimension 1  
        x = torch.arange(0, self.img_size_x_scaled, 1).float().cuda()
        y = torch.arange(0, self.img_size_y_scaled, 1).float().cuda()

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

        # reading start- and camera-orientation
        start_rotation = Quaternion(self.start_quat)
        cam_rotation = Quaternion(self.cam_quat)

        # transforming 3D points form camera- to world-frame
        for index in range(0, pointcloud_ray.shape[0]):
            rotated_point = start_rotation.rotate(cam_rotation.rotate(pointcloud_ray[index]))
            pointcloud_ray[index] = torch.tensor(rotated_point).cuda().float() + torch.tensor(self.start_tran).cuda().float()

        return pointcloud_ray, torch.tensor(faces).int().cuda() # return pointcloud and faces

    def absolute_mesh(self, xx, yy, zz):

        # reading depth, resizing it and storing it in a dataframe
        depth_df = pd.read_csv(depth_dir, sep = '\s+', header = None)
        depth_values = depth_df.values
        depth_values = resize(depth_values, (self.img_size_y_scaled, self.img_size_x_scaled), mode = 'constant')

        depth_tensor = torch.tensor(depth_values).cuda().float() # making it a torch tensor

        # 3D projection
        zz = depth_tensor / torch.sqrt(xx * xx + yy * yy + 1)
        xx = zz * xx
        yy = zz * yy

        return xx, yy, zz # return 3D points in camera-frame

    def get_depth_image(self, pointcloud_ray, faces, render_pose):
        renderer = Renderer(pointcloud_ray, faces).cuda() # initializing Renderer-object

        render_pose_SE3 = self.se3_exp(render_pose) # transforming se(3)-form to SE(3)-form

        cam_rot = torch.tensor(self.cam_quat.rotation_matrix).cuda().float().unsqueeze(dim = 0) # getting the camera-rotation matrix

        # consecutive rotation of body- and camera-frame, calculating the inverse rotation matrix
        cons_rot = torch.bmm(render_pose_SE3[:, :, :3], cam_rot)
        cons_rot_inverse = torch.inverse(cons_rot[0, :, :]).unsqueeze(dim = 0)

        # calculating inverse translation
        tran_inverse = torch.mv(cons_rot_inverse[0, :, :], torch.squeeze(render_pose_SE3[:, :, 3]))
        tran_inverse = tran_inverse.unsqueeze(dim = 0)

        # building the transformation matrix
        A = torch.zeros(3, 4).cuda().unsqueeze(dim = 0)
        B = torch.zeros(3, 4).cuda().unsqueeze(dim = 0)
        A[:, :, 3] = -1 * tran_inverse
        B[:, :, :3] = cons_rot_inverse
        T = A + B

        # generate depth-image at that position
        depth = renderer.renderer.render_depth(T, renderer.vertices, renderer.faces)
        
        # display depth image
        #test = depth
        #test = test.detach().cpu().numpy()[0][:self.img_size_y, :]
        #plt.imshow(test)
        #plt.show()

        # generating quick render image, use for testing
        #images = renderer.renderer(T, renderer.vertices, renderer.faces, torch.tanh(renderer.textures))
        #image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)[:self.img_size_y, :, :]
        #imshow(image)
        
        return depth # return depth-image


# generating reprojected and blurry images
class ImageGeneration(MeshGeneration, PoseTransformation):
    def __init__(self):
        super(ImageGeneration, self).__init__() 

    def reprojector(self, depth, cur_pose):
        # reading in reference image
        img_ref = torch.tensor(cv2.imread(img_dir, 0)).cuda().float()
        img_ref = img_ref.unsqueeze(dim = 0).unsqueeze(dim = 1).cuda().float()

        cur_pose_SE3 = self.se3_exp(cur_pose) # transforming se(3)-form to SE(3)-form

        cam_rot = torch.tensor(self.cam_quat.rotation_matrix).cuda().float().unsqueeze(dim = 0) # getting the camera-rotation matrix

        # consecutive rotation of body- and camera-frame, calculating the inverse rotation matrix
        cons_rot = torch.bmm(cur_pose_SE3[:, :, :3], cam_rot)

        # reference position
        ref_pose = self.start_pose

        ref_quat = Quaternion(ref_pose[:4]) * self.cam_quat # calculating consecutive orientation 

        # extracting translation part
        cur_tran = cur_pose_SE3[:, :, 3]
        ref_tran = ref_pose[4:]

        # generating transformation from current- to world-frame
        A = torch.zeros(4, 4).cuda().float()
        B = torch.zeros(4, 4).cuda().float()
        C = torch.zeros(4, 4).cuda().float()
        D = torch.zeros(4, 4).cuda().float()
        A[:3, :3] = cons_rot
        B[:3, 3] = cur_tran
        C[3, :3] = 0
        D[3, 3] = 1
        T_cur2W = A + B + C + D

        # generating transformation from world- to reference-frame
        T_W2ref = np.random.rand(4,4)
        T_W2ref[:3, :3] = ref_quat.inverse.rotation_matrix
        ref_tran_inv = np.matmul(-T_W2ref[:3, :3], ref_tran)
        T_W2ref[:3, 3] = ref_tran_inv
        T_W2ref[3, :3] = 0
        T_W2ref[3, 3] = 1

        T_W2ref = torch.tensor(T_W2ref).cuda().float() # creating torch tensor

        # fixing dimensions
        T_W2ref = T_W2ref.unsqueeze(dim = 0)
        T_cur2W = T_cur2W.unsqueeze(dim = 0)

        # generating transformation from current- to reference-frame
        T_cur2ref = torch.bmm(T_W2ref, T_cur2W)

        # torch tensor of dimension 1 
        x = torch.arange(0, self.img_size_x, 1).cuda().float()
        y = torch.arange(0, self.img_size_y, 1).cuda().float()

        # precalculation for 3D projection
        x_ = (x - self.K[0][2]) / self.K[0][0]
        y_ = (y - self.K[1][2]) / self.K[1][1]

        # torch tensor of dimension 2
        xx = x_.repeat(self.img_size_y, 1)
        yy = y_.view(self.img_size_y, 1).repeat(1, self.img_size_x)

        # creating depth-image tensor
        depth_tensor = depth[0, :self.img_size_y, :]
        
        depth_tensor[depth_tensor == 100] = 0

        # 3D projection
        zz = depth_tensor
        xx = zz * xx
        yy = zz * yy

        # concatenating the 3D points 
        p3d_cur = torch.stack([xx, yy, zz], dim=-1).unsqueeze(dim=0)
        Ones = torch.ones_like(p3d_cur[:, :, :, 0]).unsqueeze(dim=-1)
        p3d_cur = torch.cat([p3d_cur, Ones], dim=-1)
        p3d_cur = p3d_cur.view(1, -1, 4)
        p3d_cur = p3d_cur.transpose(2, 1)

        # changing from current- to reference-frame
        p3d_ref = T_cur2ref.bmm(p3d_cur)
        p3d_ref = p3d_ref.transpose(2, 1)

        # projecting to image-plane
        z = p3d_ref[:, :, 2].unsqueeze(dim=-1)
        xy1 = p3d_ref / z
        xy1[:, :, 0] = xy1[:, :, 0].clone() * self.K[0][0] + self.K[0][2]
        xy1[:, :, 1] = xy1[:, :, 1].clone() * self.K[1][1] + self.K[1][2]

        # normalize to interval between -1 and 1
        X = 2.0 * (xy1[:, :, 0] - self.img_size_x * 0.5 + 0.5) / (self.img_size_x - 1.)
        Y = 2.0 * (xy1[:, :, 1] - self.img_size_y * 0.5 + 0.5) / (self.img_size_y - 1.)

        # stacking up
        X = X.view(1, self.img_size_y, self.img_size_x)
        Y = Y.view(1, self.img_size_y, self.img_size_x)
        xy = torch.stack([X, Y], dim=-1)

        # grid sampling
        sample_image = torch.nn.functional.grid_sample(img_ref, xy, padding_mode='zeros')

        # changing the shape
        sample_image = sample_image.transpose(1,3).transpose(1,2)

        # for testing, display sharp reprojected image
        #test = sample_image[0, :, :, 0]
        #test = test.detach().cpu().numpy()
        #plt.imshow(test, cmap='gray')
        #plt.show()

        return torch.squeeze(sample_image) # return the reprojected image

    def blurrer(self, pointcloud_ray, faces, init_pose):
        warped_images = None # initializing variable for the warped-images

        # defining reference and current pose
        ref_pose = self.start_pose
        cur_pose = init_pose

        # extracting translation part
        ref_tran = torch.tensor(ref_pose[4:]).cuda().float()
        ref_tran = self.from_SE3t_to_se3u(ref_pose, ref_tran)
        cur_tran = cur_pose[:3]

        # extracting orientation part
        ref_rot = Quaternion(ref_pose[:4])
        ref_rot = torch.tensor(ref_rot.axis * ref_rot.angle).cuda().float()
        cur_rot = cur_pose[3:]

        for i in range(1, self.N_poses + 1):
            # calculating the timesteps at which a reprojected image should be generated
            t_i = self.cur_timestamp - self.t_exp + (i - 1) * self.t_exp / (self.N_poses - 1)
            s = (t_i - self.start_timestamp) / self.t_int

            # linearly interpolating translation and rotation
            inter_tran = ref_tran - s * (ref_tran - cur_tran)
            inter_rot = ref_rot - s * (ref_rot - cur_rot)

            # concatenating translation and rotation part
            inter_pose = torch.cat([inter_tran, inter_rot], dim = 0)

            # get depth image at the intermediate-pose
            depth = self.get_depth_image(pointcloud_ray, faces, inter_pose)

            # generate RGB-image at the intermediate-pose
            image = self.reprojector(depth, inter_pose)

            # concatenating into warped_images variable
            if i == 1:
                warped_images = image.unsqueeze(-1)
            else:
                warped_images = torch.cat([warped_images, image.unsqueeze(-1)], -1)

        blur_image = torch.mean(warped_images, -1) # taking mean to generate a blur-image
        
        # generating a grayscale-image from the RGB-image    
        #blur_image = blur_image.detach().cpu().numpy().astype(np.uint8)
        #blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
        #ImageViewer(blur_image).show()

        return blur_image # returning the generated blur-image


class Randomizer():
    def __init__(self):
        # uniformly sampling inclination and azimuth angle
        self.theta_t = random.uniform(0, math.pi)
        self.theta_r = random.uniform(0, math.pi)
        self.phi_t = random.uniform(0, 2 * math.pi)
        self.phi_r = random.uniform(0, 2 * math.pi)

    def rand_position_offset(self, dist): 
        # calculating cartesian coordinates from spherical coordinates 
        x = dist * math.sin(self.theta_t) * math.cos(self.phi_t)
        y = dist * math.sin(self.theta_t) * math.sin(self.phi_t)
        z = dist * math.cos(self.theta_t)

        return np.array([x, y, z]) # return cartesian coordinates

    def rand_rotation_offset(self, angle):
        # calculating cartesian coordinates from spherical coordinates 
        a_x = math.sin(self.theta_r) * math.cos(self.phi_r)
        a_y = math.sin(self.theta_r) * math.sin(self.phi_r)
        a_z = math.cos(self.theta_r)

        return np.array([a_x, a_y, a_z]), angle       


class Model(nn.Module, ImageGeneration):
    def __init__(self, init_pose, dist_norm, angl_norm):
        super(Model, self).__init__() 

        random_generator = Randomizer() # create random vector generator object

        tran_offset = random_generator.rand_position_offset(dist_norm)
        rot_offset, rot_offset_angle = random_generator.rand_rotation_offset(angl_norm)

        # reading quaternion, adding shift and transforming it to angle axis form
        init_pose_quat = Quaternion(init_pose[:4])
        rot_offset_quat = Quaternion(axis = rot_offset, angle = rot_offset_angle)
        init_pose_quat = init_pose_quat * rot_offset_quat
        init_pose_aa = torch.tensor(init_pose_quat.axis * init_pose_quat.angle).cuda().float()
        
        # reading in translation in SE(3) form, adding shift and transforming it to se(3) form
        init_pose_tran = init_pose[4:]
        init_pose_u_gt  = self.from_SE3t_to_se3u(init_pose[:4], torch.tensor(init_pose_tran).cuda().float())
        init_pose_tran = init_pose_tran + tran_offset
        init_pose_u  = self.from_SE3t_to_se3u(init_pose[:4], torch.tensor(init_pose_tran).cuda().float())
        print("### pose-initialization with disturbance:")
        print("    rotation:", np.concatenate([[init_pose_quat.scalar], init_pose_quat.vector], axis = 0))
        print("    translation:", init_pose_tran)

        # concatenating rotation and translation to tangent form, initializing parameter variable, printing ground-truth current pose
        self.init_pose_se3 = nn.Parameter(torch.cat([init_pose_u, init_pose_aa], dim = 0), requires_grad = True)
        print("### ground-truth initialization se(3) pose:", torch.cat([init_pose_u_gt, init_pose_aa], dim = 0))
        print("### disturbed initialization se(3) pose:", torch.cat([init_pose_u, init_pose_aa], dim = 0))

        # reeading in blurry image and converting it to torch tensor
        blur_ref = cv2.imread(blur_dir, 0)
        self.blur_ref = torch.tensor(blur_ref).cuda().float()

        self.d = 0 # parameter for image saving

    def forward(self, image_generator, pointcloud_ray, faces):
        print("### current optimized se(3) pose:", self.init_pose_se3.clone().detach()) # printing current optimized posed
        
        blur_image = image_generator.blurrer(pointcloud_ray, faces, self.init_pose_se3) # generating blurry image

        plot_blur_image = blur_image # make copy

        # plot blurry image for testing
        #cv2.imshow('image', plot_blur_image.detach().cpu().numpy().astype(np.uint8))
        #cv2.waitKey(1000)
        #v2.destroyAllWindows()

        # save intermediate blurry images
        filename = "/home/johann/motion-blur-cam-pose-tracker/semester-thesis/neural_mesh_renderer-at_assert_fix/motion-blur-pose-estimation-via-mesh-rendering/blur_%d.jpg"%self.d
        cv2.imwrite(filename, plot_blur_image.detach().cpu().numpy().astype(np.uint8))
        self.d = self.d + 1

        loss = torch.sum((blur_image - self.blur_ref) * (blur_image - self.blur_ref)) # loss function, sum of quadratic deviation
        
        # calculating current translation
        iter_pose_SE3 = self.se3_exp(self.init_pose_se3)
        iter_tran = iter_pose_SE3[0,:,3].detach().cpu().numpy()

        # show difference image
        #difference_image = blur_image - self.blur_ref # calculate difference image
        #plot_difference_image = difference_image
        #cv2.imshow('image', plot_difference_image.detach().cpu().numpy().astype(np.uint8))
        #cv2.waitKey(1000)
        #cv2.destroyAllWindows()

        return loss, iter_tran # return loss and current translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', type = int, default = 1)
    parser.add_argument('-g', '--gpu', type=int, default=1)
    args = parser.parse_args()

    # set default tensor
    if args.gpu > 0:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("### GPU setting:", bool(args.gpu))

    # hyperparameters definitions
    init_pose = np.array([0.951512, 0.0225991, 0.0716038, -0.298306, -0.821577, 1.31002, 0.911207])
    dist_norm = 0.5
    angl_norm = 0

    print("### pose-initialization:")
    print("    rotation:", init_pose[:4])
    print("    translation:", init_pose[4:])
    print("    translation-disturbance:", dist_norm)
    print("    rotation-disturbance:", angl_norm)

    # initializing class objects
    room_mesh = MeshGeneration()
    model = Model(init_pose, dist_norm, angl_norm).cuda()
    image_generator = ImageGeneration()
    #random_generator = Randomizer()
    print("### all objects initialized")

    # generating pointcloud and mesh
    pointcloud_ray, faces = room_mesh.generate_mean_mesh()
    print("### pointcloud and polygon-mesh generated at scale:", room_mesh.scale)

    # save pointcloud and mesh option
    np.savetxt('pointcloud.txt', pointcloud_ray)
    meshio.write_points_cells("polygon_mesh.off", pointcloud_ray, {"triangle": faces})
    print("### pointcloud and polygon-mesh saved for MeshLab analysis: <pointcloud.txt> and <polygon_mesh.off>")

    print("### start generating artificial blur images")

    # testing purpose
    #loss = model.forward(image_generator, pointcloud_ray, faces) # test one iteration

    # testing pose transformation functions
    pose_tester = PoseTransformation()
    #test_pose = np.array([0.986339, -0.0150744, 0.0960326, 0.132985, -1.10328, -0.498209, 0.909185])
    #quat_SE3 = Quaternion(test_pose[:4])
    #rot_mat_SE3 = quat_SE3.rotation_matrix
    #angle_axis_se3 = quat_SE3.axis * quat_SE3.angle
    #print("Rotation Matrix SE(3):")
    #print(rot_mat_SE3)
    #print("Angle Axis se(3) (true value):", angle_axis_se3)
    #print("Translation SE(3):", test_pose[4:])
    #u_se3 = pose_tester.from_SE3t_to_se3u(test_pose[:4], torch.tensor(test_pose[4:]).cuda().float())
    #print("Translation se(3): ", u_se3)
    #SE_3 = pose_tester.se3_exp(torch.tensor(np.concatenate((u_se3.cpu().numpy(), angle_axis_se3), axis = 0)).cuda().float())
    #print("Results from exponential mapping:")
    #print(SE_3)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) # optimizer, tuning needed

    rounds = 1000
    loop = tqdm(range(rounds))

    # loop optimization
    for i in loop:
        optimizer.zero_grad() # set gradients to zero 
        
        # calculating loss function and backpropagating
        loss, iter_tran = model.forward(image_generator, pointcloud_ray, faces)
        loss.backward()
        optimizer.step()

        loop.set_description("### optimizing (loss %.4f)" % loss.data) # loss print

        # calculate and print distance error
        distance_error = math.sqrt((iter_tran[0]-init_pose[4]) ** 2 + (iter_tran[1]-init_pose[5]) ** 2 + (iter_tran[2]-init_pose[6]) ** 2)
        print("### current translational error:", distance_error)

        # break condition
        if loss.item() < 200000000.:
            loop.close()
            break

    print("### pipeline completed")


if __name__ == '__main__':
    main()
