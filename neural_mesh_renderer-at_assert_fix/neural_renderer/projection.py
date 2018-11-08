from __future__ import division

import torch


def projection(vertices, P, K, dist_coeffs, img_size):
    '''
    Calculate projective transformation of vertices given a projection matrix
    P: 3x4 transformation matrix [r, t], from world to camera view
    K: camera intrinsics [fx fy cx cy]
    dist_coeffs: vector of distortion coefficients
    img_size: image resolution [W, H] or [X, Y]
    '''
    vertices = torch.cat([vertices, torch.ones_like(vertices[:, :, None, 0])], dim=-1)
    vertices = torch.bmm(vertices, P.transpose(2, 1))
    if vertices.dim() == 2:
        vertices = torch.unsqueeze(vertices, 0)

    x = vertices[:, :, 0] 
    y = vertices[:, :, 1] 
    z = vertices[:, :, 2]

    x_ = x / (z + 1e-5)
    y_ = y / (z + 1e-5)
        
    ## Get distortion coefficients from vector
    #k1 = dist_coeffs[:, 0]
    #k2 = dist_coeffs[:, 1]
    #p1 = dist_coeffs[:, 2]
    #p2 = dist_coeffs[:, 3]
    #k3 = dist_coeffs[:, 4]

    ## we use x_ for x' and x__ for x'' etc.
    #r = torch.sqrt(x_ ** 2 + y_ ** 2)
    
    #x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
    #y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_

    # get camera intrinsics
    fx = K[0]
    fy = K[1]
    cx = K[2]
    cy = K[3]

    X = img_size[:, 0].float()
    Y = img_size[:, 1].float()

    a = 2. * fx / X
    b = (2. * cx - X + 1) / X

    c = 2. * fy / Y
    d = (2. * cy - Y + 1) / Y

    x__ = a * x_ + b
    y__ = c * y_ + d
    
    vertices = torch.stack([x__, y__, z], dim=-1)
    return vertices
