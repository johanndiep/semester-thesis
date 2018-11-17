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
        pass



def make_reference_image(filename_ref, filename_obj):
    pass




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example5_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example5_result.gif'))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=1)
    parser.add_argument('-g', '--gpu', type=int, default=1)
    args = parser.parse_args()

    if args.gpu > 0:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.make_reference_image:
        make_reference_image(args.filename_ref, args.filename_obj)

   # model = Model(args.filename_obj, args.filename_ref)
   # model.cuda()


if __name__ == '__main__':
    main()
