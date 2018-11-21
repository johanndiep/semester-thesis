#Semester Thesis: Implementation of the Motion-blur Aware Camera-Pose Estimation Algorithm in Python via Mesh Rendering
#Author: Johann Diep (jdiep@student.ethz.ch)

import argparse
import torch
import torch.nn as nn
import os
import meshzoo
import numpy as np
import sys
import neural_renderer as nr
import OpenEXR as exr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

class CameraParameter():
    def __init__(self):
        self.K = torch.tensor([[320., 0, 320.], [0., 320., 240.], [0., 0., 1.]]).float().cuda()

        self.img_size_x = 480
        self.img_size_y = 640

        self.scale = 0

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

        unit_ray = torch.stack([xx, yy, zz], dim=-1)
        unit_ray = unit_ray.view(-1, 3)

        return unit_ray, faces

    def absolute_mesh(self, unit_ray):
        absolute_mesh = torch.ones_like(unit_ray)

        for ray_index in range(0, len(unit_ray)):
            print("Hello")

        return absolute_mesh

    def make_reference_image(self):
        pass

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
    unit_ray, faces = room_mesh.generate_mean_mesh()

    if args.make_reference_image:
        room_mesh.make_reference_image()

    print("Everything ok")

if __name__ == '__main__':
    main()
