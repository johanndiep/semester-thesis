# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file sets up the depth renderer, reprojector and the blurrer.


# libraries
from lib import dataset
from lib import posetransformation
from lib import renderer

# external libraries
import neural_renderer as nr
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from pyquaternion import Quaternion


torch.set_default_tensor_type(torch.cuda.FloatTensor) # using CUDA


# generating depth, reprojected and blurry images
class ImageGeneration(posetransformation.PoseTransformation, dataset.Extrinsics, dataset.Intrinsics, dataset.Sharp, dataset.GroundTruth, dataset.ImageLogs):
    def __init__(self):
        super(ImageGeneration, self).__init__()

    # render a depth image at an arbitrary position
    def render_depth(self, cam_index, pointcloud_ray, faces, render_pose, pyramid_scale):
        renderer_obj = renderer.Renderer(cam_index, pointcloud_ray, faces, pyramid_scale) # initialize renderer object

        render_pose_SE3  = self.se3_exp(render_pose) # transforming se(3)-form to SE(3)-form

        # get camera rotation and translation
        cam_quat = Quaternion(self.get_extrinsics(cam_index)[0])
        cam_tran_SE3 = torch.tensor(self.get_extrinsics(cam_index)[1])

        cam_rot_SE3 = torch.tensor(cam_quat.rotation_matrix).float().unsqueeze(dim = 0) # getting the camera-rotation matrix

        # consecutive rotation of body- and camera-frame, calculating the inverse rotation matrix
        cons_rot_SE3 = torch.bmm(render_pose_SE3[:, :, :3], cam_rot_SE3)
        cons_rot_inverse_SE3 = torch.inverse(cons_rot_SE3[0, :, :]).unsqueeze(dim = 0)

        cons_tran_SE3 = torch.mv(render_pose_SE3[0, :, :3], cam_tran_SE3) + torch.squeeze(render_pose_SE3[:, :, 3]) # consecutive translation

        # calculating inverse translation
        cons_tran_inverse_SE3 = -1 * torch.mv(cons_rot_inverse_SE3[0, :, :], cons_tran_SE3)
        cons_tran_inverse_SE3 = cons_tran_inverse_SE3.unsqueeze(dim = 0)

        # building the transformation matrix
        A = torch.zeros(3, 4).unsqueeze(dim = 0)
        B = torch.zeros(3, 4).unsqueeze(dim = 0)
        A[:, :, 3] = cons_tran_inverse_SE3
        B[:, :, :3] = cons_rot_inverse_SE3
        T = A + B

        # generate depth-image at that position
        depth = renderer_obj.renderer.render_depth(T, renderer_obj.vertices, renderer_obj.faces)
        
        # display depth image
        # test = depth
        # test = test.detach().cpu().numpy()[0][:self.get_intrinsics(cam_index)[1], :]
        # plt.imshow(test)
        # plt.show()

        # generating quick render image, use for testing
        # images = renderer_obj.renderer(T, renderer_obj.vertices, renderer_obj.faces, torch.tanh(renderer_obj.textures))
        # image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)[:self.get_intrinsics(cam_index)[1], :, :]
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        return depth # return depth-image

    # generate a sharp image at an arbitrary position
    def reprojector(self, cam_index, img_ref, t_ref, depth, render_pose, pyramid_scale):
    	# calibration and resolution read
        K = torch.tensor(self.get_scaled_intrinsics(cam_index, pyramid_scale)[2]).float()
        img_size_x = self.get_scaled_intrinsics(cam_index, pyramid_scale)[0]
        img_size_y = self.get_scaled_intrinsics(cam_index, pyramid_scale)[1]

        # reading in reference image
        img_ref = torch.tensor(self.get_scaled_sharp_image(cam_index, img_ref, pyramid_scale)).float()
        img_ref = img_ref.unsqueeze(dim = 0).unsqueeze(dim = 1).float()

        render_pose_SE3 = self.se3_exp(render_pose) # transforming se(3)-form to SE(3)-form

        # getting the camera-rotation matrix and translation
        cam_quat = Quaternion(self.get_extrinsics(cam_index)[0])
        cam_rot_SE3 = torch.tensor(cam_quat.rotation_matrix).float().unsqueeze(dim = 0)
        cam_tran_SE3 = torch.tensor(self.get_extrinsics(cam_index)[1]).float()

        # consecutive rotation and translation of body- and camera-frame
        cons_rot_SE3 = torch.bmm(render_pose_SE3[:, :, :3], cam_rot_SE3)
        render_tran_SE3 = render_pose_SE3[:, :, 3]
        cons_tran_SE3 = torch.mv(render_pose_SE3[0, :, :3], cam_tran_SE3) + torch.squeeze(render_tran_SE3)

        # reference pose, calculating consecutive orientation and translation
        ref_quat, ref_tran_SE3 = self.get_pose_at(t_ref)
        cons_ref_quat = Quaternion(ref_quat) * cam_quat
        cons_ref_tran_SE3 = np.matmul(Quaternion(ref_quat).rotation_matrix, cam_tran_SE3) + ref_tran_SE3

        # generating transformation from current- to world-frame
        A = torch.zeros(4, 4).float()
        B = torch.zeros(4, 4).float()
        C = torch.zeros(4, 4).float()
        D = torch.zeros(4, 4).float()
        A[:3, :3] = cons_rot_SE3
        B[:3, 3] = cons_tran_SE3
        C[3, :3] = 0
        D[3, 3] = 1
        T_cur2W = A + B + C + D

        # generating transformation from world- to reference-frame
        T_W2ref = np.random.rand(4,4)
        T_W2ref[:3, :3] = cons_ref_quat.inverse.rotation_matrix
        T_W2ref[:3, 3] = np.matmul(-T_W2ref[:3, :3], cons_ref_tran_SE3)
        T_W2ref[3, :3] = 0
        T_W2ref[3, 3] = 1

        T_W2ref = torch.tensor(T_W2ref).float() # creating torch tensor

        # fixing dimensions
        T_W2ref = T_W2ref.unsqueeze(dim = 0)
        T_cur2W = T_cur2W.unsqueeze(dim = 0)

        # generating transformation from current- to reference-frame
        T_cur2ref = torch.bmm(T_W2ref, T_cur2W)

        # torch tensor of dimension 1 
        x = torch.arange(0, img_size_x, 1).float()
        y = torch.arange(0, img_size_y, 1).float()

        # precalculation for 3D projection
        x_ = (x - K[0][2]) / K[0][0]
        y_ = (y - K[1][2]) / K[1][1]

        # torch tensor of dimension 2
        xx = x_.repeat(img_size_y, 1)
        yy = y_.view(img_size_y, 1).repeat(1, img_size_x)

        # creating depth-image tensor
        depth_tensor = depth[0, :img_size_y, :]
 
        depth_tensor[depth_tensor == 100] = -100 # avoiding same plane backprojection

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
        xy1[:, :, 0] = xy1[:, :, 0].clone() * K[0][0] + K[0][2]
        xy1[:, :, 1] = xy1[:, :, 1].clone() * K[1][1] + K[1][2]

        # normalize to interval between -1 and 1
        X = 2.0 * (xy1[:, :, 0] - img_size_x * 0.5 + 0.5) / (img_size_x - 1.)
        Y = 2.0 * (xy1[:, :, 1] - img_size_y * 0.5 + 0.5) / (img_size_y - 1.)

        # stacking up
        X = X.view(1, img_size_y, img_size_x)
        Y = Y.view(1, img_size_y, img_size_x)
        xy = torch.stack([X, Y], dim=-1)

        # grid sampling
        sample_image = torch.nn.functional.grid_sample(img_ref, xy, padding_mode='zeros')

        # changing the shape
        sample_image = sample_image.transpose(1,3).transpose(1,2)

        # for testing, display sharp reprojected image
        # test = sample_image[0, :, :, 0]
        # test = test.detach().cpu().numpy()
        # plt.imshow(test, cmap='gray')
        # plt.show()

        return torch.squeeze(sample_image) # return the reprojected image

    # generate a blur image at an arbitrary position
    def blurrer(self, cam_index, img_ref, img_cur, t_ref, t_cur, pointcloud_ray, faces, inter_pose, N_poses, pyramid_scale):
        warped_images = None # initializing variable for the warped-images

        # defining reference and current pose
        ref_quat, ref_tran_SE3 = self.get_pose_at(t_ref)

        # extracting translation part
        ref_tran_SE3 = torch.tensor(ref_tran_SE3).float()
        ref_tran_se3 = self.from_SE3t_to_se3u(Quaternion(ref_quat), ref_tran_SE3)
        inter_tran_se3 = inter_pose[:3]

        # extracting orientation part
        ref_quat = Quaternion(ref_quat)
        ref_aa = torch.tensor(ref_quat.axis * ref_quat.angle).float()
        inter_aa = inter_pose[3:]

        for i in range(1, N_poses + 1):
            # calculating the timesteps at which a reprojected image should be generated
            t_i = t_cur - self.get_timestamp(cam_index, img_cur)[1] + (i - 1) * self.get_timestamp(cam_index, img_cur)[1] / (N_poses - 1)
            s = (t_i - t_ref) / (t_cur - t_ref)

            # linearly interpolating translation and rotation
            i_tran_se3 = ref_tran_se3 - s * (ref_tran_se3 - inter_tran_se3)
            i_aa = ref_aa - s * (ref_aa - inter_aa)

            # concatenating translation and rotation part
            i_pose = torch.cat([i_tran_se3, i_aa], dim = 0)

            # get depth image at the intermediate-pose
            depth = self.render_depth(cam_index, pointcloud_ray, faces, i_pose, pyramid_scale)

            # generate RGB-image at the intermediate-pose
            image = self.reprojector(cam_index, img_ref, t_ref, depth, i_pose, pyramid_scale)

            # concatenating into warped_images variable
            if i == 1:
                warped_images = image.unsqueeze(-1)
            else:
                warped_images = torch.cat([warped_images, image.unsqueeze(-1)], -1)

        blur_image = torch.mean(warped_images, -1) # taking mean to generate a blur-image

        return blur_image # returning the generated blur-image