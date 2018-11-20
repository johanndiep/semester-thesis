#Semester Thesis: Implementation of the Motion-blur Aware Camera-Pose Estimation Algorithm in Python via Mesh Rendering
#Author: Johann Diep (jdiep@student.ethz.ch)

import argparse
import torch
import torch.nn as nn
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref=None):
        super(Model, self).__init__()

        K = torch.tensor([[320., 0, 320.], [0., 320., 240.], [0., 0., 1.]]).float().cuda()
        self.K = torch.unsqueeze(K, 0)





def make_mesh(filename_ref, filename_obj):
    model = Model(filename_obj)
    unit_ray, faces = generate_mean_mesh(model.K, model.img_size, scale = 0)


def generate_mean_mesh(K, img_size, scale = 0):
    scale = 2 ** scale




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'room_mesh.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example5_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example5_result.gif'))
    parser.add_argument('-mr', '--generate_mesh', type=int, default=1)
    parser.add_argument('-g', '--gpu', type=int, default=1)
    args = parser.parse_args()

    if args.gpu > 0:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.generate_mesh:
        make_mesh(args.filename_ref, args.filename_obj)

   # model = Model(args.filename_obj, args.filename_ref)
   # model.cuda()


if __name__ == '__main__':
    main()
