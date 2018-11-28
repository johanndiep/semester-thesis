#Semester Thesis: Implementation of the Motion-blur Aware Camera-Pose Estimation Algorithm in Python via Mesh Rendering
#Author: Johann Diep (jdiep@student.ethz.ch)

import argparse
import torch
import torch.nn as nn
import os
import meshzoo
import meshio
import numpy as np
import sys
import neural_renderer as nr
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from skimage.io import imsave, imread
from scipy.misc import imshow
from pyquaternion import Quaternion


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
depth_dir = '/home/johann/motion-blur-cam-pose-tracker/semester-thesis/RelisticRendering-dataset/depth/cam0/depth_map.csv'


class CameraParameter():
    def __init__(self):
        super(CameraParameter, self).__init__() 

        self.K = torch.tensor([[320., 0, 320.], [0., 320., 240.], [0., 0., 1.]]).float().cuda()

        self.img_size_x = 640
        self.img_size_y = 480

        self.scale = 0

        self.dist_coeffs = torch.tensor([0, 0, 0, 0, 0]).float().cuda()

        start_pose = np.array([0.93579, 0.0281295, 0.0740478, -0.343544, -0.684809, 1.59021, 0.91045])
        self.start_quat = Quaternion(start_pose[:4])
        self.start_tran = start_pose[4:] 
        
        cam_pose = np.array([0.5, -0.5, 0.5, -0.5, 0, 0, 0])
        self.cam_quat = Quaternion(cam_pose[:4])
        self.cam_tran = cam_pose[4:] 

class PoseTransformation():
    def se3_exp(self, tangent):
        if tangent.dim() < 2:
            tangent = tangent.unsqueeze(dim = 0)

        t = tangent[:, :3]
        phi = tangent[:, 3:]

        R, R_jac = self.so3_exp(phi)
        t = t.unsqueeze(2)

        trans = torch.bmm(R_jac, t)
        return torch.cat([R, trans], dim = 2)

    def so3_exp(self, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim = 0)

        jac = phi.__class__(phi.shape[0], 3, 3)
        angle = phi.norm(p = 2, dim = 1)
        angle = angle + 1e-5

        s = angle.sin()
        c = angle.cos()

        s_div_angle = s / angle
        one_minus_s_div_angle = 1. - s_div_angle
        one_minus_c = 1. - c
        one_minus_c_div_angle = one_minus_c / angle

        angle = angle.unsqueeze(1)
        axis = phi / angle

        I = torch.eye(3).expand_as(jac)

        s_div_angle = s_div_angle.unsqueeze(1).unsqueeze(2)
        one_minus_s_div_angle = one_minus_s_div_angle.unsqueeze(1).unsqueeze(2)
        one_minus_c_div_angle = one_minus_c_div_angle.unsqueeze(1).unsqueeze(2)

        axis2 = self.outer(axis, axis)
        wedge_axis = self.wedge(axis)

        A_jac = s_div_angle * I
        B_jac = one_minus_s_div_angle * axis2
        C_jac = one_minus_c_div_angle * wedge_axis

        c = c.unsqueeze(1).unsqueeze(2)
        s = s.unsqueeze(1).unsqueeze(2)
        one_minus_c = one_minus_c.unsqueeze(1).unsqueeze(2)

        A = c * I
        B = one_minus_c * axis2
        C = s * wedge_axis

        return A + B + C, A_jac + B_jac + C_jac

    def outer(self, vecs1, vecs2):
        if vecs1.dim() < 2:
            vecs1 = vecs1.unsqueeze(dim = 0)

        if vecs2.dim() < 2:
            vecs2 = vecs2.unsqueeze(dim = 0)

        if vecs1.shape[0] != vecs2.shape[0]:
            raise ValueError("Inconsistent batch sizes {} and {}".format(vecs1.shape[0], vec2.shape[0]))

        return torch.bmm(vecs1.unsqueeze(dim = 2), vecs2.unsqueeze(dim = 2).transpose(2, 1)).squeeze_()

    def wedge(self, phi):
        if phi.dim() < 2:
            phi.unsqueeze(dim = 0)

        if phi.shape[1] != 3:
            raise ValueError("phi must have shape ({},) or (N,{})".format(3, 3))

        Phi = phi.__class__(phi.shape[0], 3, 3).zero_()
        Phi[:, 0, 1] = -phi [:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]

        return Phi.squeeze_()


class Model(CameraParameter, nn.Module, PoseTransformation):
    def __init__(self, vertices, faces, filename_ref = None):
        super(Model, self).__init__()
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3,
                              dtype = torch.float32)
        self.register_buffer('textures', textures)

        K = self.K
        K = torch.unsqueeze(K, 0)

        renderer = nr.ProjectiveRenderer(image_size = [self.img_size_x, self.img_size_y],
                                                       K = K)
        self.renderer = renderer

        if filename_ref != None:
            image_ref  = torch.from_numpy((imread(filename_ref).max(-1) != 0).astype(np.float32))
            self.register_buffer('image_ref', image_ref)

            self.T = nn.Parameter(torch.from_numpy(np.array([1.0, 2.0, 3.0, 1, 0, 0], dtype = np.float32)).float().cuda())

    def forward(self):
        M = self.se3_exp(self.T)
        image = self.renderer(M, self.vertices, self.faces, mode = 'silhouettes')
        loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)
        return loss


class MeshGeneration(CameraParameter):
    def __init__(self):
        super(MeshGeneration, self).__init__()

        self.scale_power = 2 ** self.scale
        self.K = self.K / self.scale_power

        self.img_size_x = int(self.img_size_x // self.scale_power)
        self.img_size_y = int(self.img_size_y // self.scale_power)

    def generate_mean_mesh(self):
        _, faces = meshzoo.rectangle(xmin = -1, xmax = 1,
                                     ymin = -1, ymax = 1.,
                                     nx = self.img_size_x, ny = self.img_size_y,
                                     zigzag = True)

        x = torch.arange(0, self.img_size_x, 1).float().cuda()
        y = torch.arange(0, self.img_size_y, 1).float().cuda()

        x_ = (x - self.K[0][2]) / self.K[0][0]
        y_ = (y - self.K[1][2]) / self.K[1][1]

        xx = x_.repeat(self.img_size_y, 1)
        yy = y_.view(self.img_size_y, 1).repeat(1, self.img_size_x)
        zz = torch.ones_like(xx)

        xx, yy, zz = absolute_mesh = self.absolute_mesh(xx, yy, zz)

        pointcloud_ray = torch.stack([xx, yy, zz], dim=-1)
        pointcloud_ray = pointcloud_ray.view(-1, 3)

        start_rotation = Quaternion(self.start_quat)
        cam_rotation = Quaternion(self.cam_quat)

        for index in range(0, pointcloud_ray.shape[0]):
            rotated_point = start_rotation.rotate(cam_rotation.rotate(pointcloud_ray[index]))
            pointcloud_ray[index] = torch.tensor(rotated_point).float() + torch.tensor(self.start_tran).float()

        print(pointcloud_ray)

        return pointcloud_ray, torch.tensor(faces).int().cuda()

    def absolute_mesh(self, xx, yy, zz):

        depth_df = pd.read_csv(depth_dir, sep = '\s+', header = None)
        depth_values = depth_df.values

        depth_tensor = torch.tensor(depth_values).float().cuda()

        zz = depth_tensor / torch.sqrt(xx ** 2 + yy ** 2 + 1)
        xx = zz * xx
        yy = zz * yy

        return xx, yy, zz

    def make_reference_image(self, filename_ref, pointcloud_ray, faces):
        model = Model(pointcloud_ray, faces)
        model.cuda()

        transformation = PoseTransformation()
        T = transformation.se3_exp(torch.tensor([[-0.8584986, 1.8585109, -1.5835546, -0.684809, 1.59021, 0.91045]]))

        images = model.renderer(T, model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
        imsave(filename_ref, image)
        imshow(image)

        #depth = model.renderer.render_depth(T, model.vertices, model.faces)
        #depth = depth.detach().cpu().numpy()[0]
        #np.savetxt('depth.txt', depth, delimiter = ' ')


def main():
    #print(sys.version)
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'room_mesh.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example5_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example5_result.gif'))
    parser.add_argument('-mri', '--make_reference_image', type = int, default = 1)
    parser.add_argument('-g', '--gpu', type=int, default=1)
    args = parser.parse_args()

    if args.gpu > 0:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    room_mesh = MeshGeneration()
    pointcloud_ray, faces = room_mesh.generate_mean_mesh()   

    np.savetxt('pointcloud.txt', pointcloud_ray)
    meshio.write_points_cells("room_mesh.off", pointcloud_ray, {"triangle": faces})

    if args.make_reference_image:
        room_mesh.make_reference_image(args.filename_ref, pointcloud_ray, faces)

    # model = Model(pointcloud_ray, faces, args.filename_ref)
    # model.cuda()

    # transformation = PoseTransformation()

    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

    # for i in tqdm.tqdm(range(1000)):
    #     optimizer.zero_grad()
    #     loss = model.forward()
    #     loss.backward()
    #     optimizer.step()

    #     print (model.T)

    #     M = transformation.se3_exp(model.T)
    #     images = model.renderer(M, model.vertices, model.faces, torch.tanh(model.textures))
    #     image = images.detach().cpu().numpy()[0].transpose(1,2,0)
    #     imsave('/tmp/_tmp_%04d.png' % i, image)
    #     loop.set_description('Optimizing (loss %.4f)' % loss.data)
    #     if loss.item() < 70:
    #         break

    print("Everything ok")


if __name__ == '__main__':
    main()
