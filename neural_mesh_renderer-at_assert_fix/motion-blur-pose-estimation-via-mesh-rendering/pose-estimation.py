
#Semester Thesis: Implementation of the Motion-blur Aware Camera-Pose Estimation Algorithm in Python via Mesh Rendering
#Author: Johann Diep (jdiep@student.ethz.ch)

import argparse
import torch
import torch.nn as nn
import os
import meshzoo
import numpy as np
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class CameraParameter():
    def __init__(self, filename_obj, filename_ref=None):
        pass



def generate_mean_mesh(img_size_x, img_size_y, K, scale = 0):
    scale = 2 ** scale
    K = K / scale
    img_size_x = int(model.img_size_x // scale)
    img_size_y = int(model.img_size_y // scale)

    _, faces = meshzoo.rectangle(xmin = -1, xmax = 1,
                                 ymin = -1, ymax = 1.,
                                 nx = img_size_x, ny = img_size_y,
                                 zigzag = True)

    x = torch.arange(0, img_size_x, 1).float().cuda()
    y = torch.arange(0, img_size_y, 1).float().cuda()

    x_ = (x - K[0][2]) / K[0][0]
    y_ = (y - K[1][2]) / K[1][1]

    return x_, y_


def main():
    #print(sys.version)
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'room_mesh.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example5_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example5_result.gif'))
    parser.add_argument('-mr', '--generate_mesh', type=int, default=1)
    parser.add_argument('-g', '--gpu', type=int, default=1)
    args = parser.parse_args()

    if args.gpu > 0:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    K = torch.tensor([[320., 0, 320.], [0., 320., 240.], [0., 0., 1.]]).float().cuda()
    K = torch.unsqueeze(K, 0)

    img_size_x = 480
    img_size_y =  640

    if args.generate_mesh:
        a,b = generate_mean_mesh(img_size_x, img_size_y, K, scale = 0)

    print(a)

if __name__ == '__main__':
    main()
