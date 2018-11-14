"""
Example 4. Finding camera parameters.
"""
import os
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
from scipy.misc import imshow

from liegroups.torch import SE3
from liegroups.torch import SO3

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def outer(vecs1, vecs2):
    """Return the N x D x D outer products of a N x D batch of vectors,
    or return the D x D outer product of two D-dimensional vectors.
    """
    # Default batch size is 1
    if vecs1.dim() < 2:
        vecs1 = vecs1.unsqueeze(dim=0)

    if vecs2.dim() < 2:
        vecs2 = vecs2.unsqueeze(dim=0)

    if vecs1.shape[0] != vecs2.shape[0]:
        raise ValueError("Got inconsistent batch sizes {} and {}".format(
            vecs1.shape[0], vecs2.shape[0]))

    return torch.bmm(vecs1.unsqueeze(dim=2),
                     vecs2.unsqueeze(dim=2).transpose(2, 1)).squeeze_()

def wedge(phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=0)

        if phi.shape[1] != 3:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(3, 3))

        Phi = phi.__class__(phi.shape[0], 3, 3).zero_()
        Phi[:, 0, 1] = -phi[:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]
        return Phi.squeeze_()
    
def so3_exp(phi):
    if phi.dim() < 2:
        phi = phi.unsqueeze(dim=0)

    jac = phi.__class__(phi.shape[0], 3, 3)
    angle = phi.norm(p=2, dim=1)
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
    
    axis2 = outer(axis, axis)
    wedge_axis = wedge(axis)
    
    A_jac = s_div_angle * I
    B_jac = one_minus_s_div_angle * axis2
    C_jac = one_minus_c_div_angle * wedge_axis
    
    c = c.unsqueeze(1).unsqueeze(2)
    s = s.unsqueeze(1).unsqueeze(2)
    one_minus_c = one_minus_c.unsqueeze(1).unsqueeze(2)
    
    A = c* I 
    B = one_minus_c * axis2
    C = s * wedge_axis
    
    return A + B + C, A_jac + B_jac + C_jac

def se3_exp(tangent):
    if tangent.dim() < 2:
        tangent = tangent.unsqueeze(dim=0)
        
    t = tangent[:, :3]
    phi = tangent[:, 3:]
    
    R, R_jac = so3_exp(phi)
    t = t.unsqueeze(2)
    
    trans = torch.bmm(R_jac, t)
    return torch.cat([R, trans], dim=2)

class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref=None):
        super(Model, self).__init__()
        # load .obj
        vertices, faces = nr.load_obj(filename_obj)
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        print(vertices.shape)

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # setup renderer
        renderer = nr.ProjectiveRenderer()
        self.renderer = renderer

        if filename_ref:
            # load reference image
            image_ref = torch.from_numpy((imread(filename_ref).max(-1) != 0).astype(np.float32))
            self.register_buffer('image_ref', image_ref)

            # camera parameters
            self.T = nn.Parameter(torch.from_numpy(np.array([1.0, 2.0, 3.0, 1, 0, 0], dtype=np.float32)).float().cuda())
            
            K = torch.tensor([[128., 0., 128.], [0., 128., 128.], [0., 0., 1.]]).float().cuda()
            K = torch.unsqueeze(K, 0)
            self.renderer.K = K

            dist_coeffs = torch.tensor([0, 0, 0, 0, 0]).float().cuda()
            dist_coeffs = torch.unsqueeze(dist_coeffs, 0)
            self.renderer.dist_coeffs = dist_coeffs

    def forward(self):
        M = se3_exp(self.T)
        image = self.renderer(M, self.vertices, self.faces, mode='silhouettes')
        loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)
        return loss

def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()


def make_reference_image(filename_ref, filename_obj):
    model = Model(filename_obj)
    model.cuda()

    T = se3_exp(torch.tensor([[0.0, 0.0, 3.0, 0, 0, 0]]))
    #print(T)
    
    K = torch.tensor([[128., 0., 128.], [0., 128., 128.], [0., 0., 1.]]).float().cuda()
    K = torch.unsqueeze(K, 0)
    model.renderer.K = K

    dist_coeffs = torch.tensor([0, 0, 0, 0, 0]).float().cuda()
    dist_coeffs = torch.unsqueeze(dist_coeffs, 0)
    model.renderer.dist_coeffs = dist_coeffs

    images = model.renderer(T, model.vertices, model.faces, torch.tanh(model.textures))
    #print(images.size())
    image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)    
    imsave(filename_ref, image)
    #imshow(image)

    depth = model.renderer.render_depth(T, model.vertices, model.faces)
    depth = depth.detach().cpu().numpy()[0]
    #print(depth.shape)
    #print(depth[128, 128])

    np.savetxt('depth.txt', depth, delimiter=' ') 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example4_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example4_result.gif'))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=1)
    args = parser.parse_args()
    
    if args.gpu > 0:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.make_reference_image:
        make_reference_image(args.filename_ref, args.filename_obj)
        return 

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()

    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loop = tqdm.tqdm(range(1000))
    for i in loop:
        optimizer.zero_grad()
        loss = model.forward()
        loss.backward()

        #print('\n')
        #for param in model.parameters():
        #    print(param.data, param.grad, param.size())

        optimizer.step()

        print(model.T)

        M = se3_exp(model.T)
        images = model.renderer(M, model.vertices, model.faces, torch.tanh(model.textures))

        image = images.detach().cpu().numpy()[0].transpose(1,2,0)
        imsave('/tmp/_tmp_%04d.png' % i, image)
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        if loss.item() < 70:
            break
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
