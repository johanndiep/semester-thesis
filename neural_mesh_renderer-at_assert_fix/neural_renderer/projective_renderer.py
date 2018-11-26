from __future__ import division
import math

import torch
import torch.nn as nn
import numpy
import neural_renderer as nr

class ProjectiveRenderer(nn.Module):
    def __init__(self, image_size=256, anti_aliasing=False, background_color=[0,0,0],
                 fill_back=False,
                 dist_coeffs=None, K=None,
                 near=0.1, far=100,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,1,0]):
        '''
        Calculate projective transformation of vertices given a projection matrix
        T: Bx1 SE3 object
        K: Bx3x3 camera intrinsics
        dist_coeffs: Bx5 vector of distortion coefficients [k1, k2, p1, p2, k3]
        img_size: Bx2 image resolution [W, H] or [X, Y]
        '''
        super(ProjectiveRenderer, self).__init__()

        # rendering
        self.image_size = torch.unsqueeze(torch.tensor([image_size, image_size]).cuda(), 0)
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back
        
        # for projective camera with P = [t, R], K and dist_coeffs
        self.K = K

        if dist_coeffs == None:
            self.dist_coeffs = torch.tensor([0, 0, 0, 0, 0]).float().cuda().unsqueeze(0)

        self.near = near
        self.far = far

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction 

        # rasterization
        self.rasterizer_eps = 1e-3

    def forward(self, T, vertices, faces, textures=None, mode=None):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''
        if mode is None:
            return self.render(T, vertices, faces, textures)
        elif mode == 'silhouettes':
            return self.render_silhouettes(T, vertices, faces)
        elif mode == 'depth':
            return self.render_depth(T, vertices, faces)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, T, vertices, faces):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        vertices = nr.projection(vertices, T, self.K, self.dist_coeffs,  self.image_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size[0, 0], self.anti_aliasing)
        return images

    def render_depth(self, T, vertices, faces):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        vertices = nr.projection(vertices, T, self.K, self.dist_coeffs,  self.image_size)
            
        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size[0, 0], self.anti_aliasing)
        return images

    def render(self, T, vertices, faces, textures):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        vertices = nr.projection(vertices, T, self.K, self.dist_coeffs,  self.image_size)
        
        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(faces, textures, self.image_size[0, 0], self.anti_aliasing, self.near, self.far, self.rasterizer_eps, self.background_color)
        return images
