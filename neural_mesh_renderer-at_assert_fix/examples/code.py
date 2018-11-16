import torch
import numpy as np
import torch.nn as nn
import neural_renderer as nr

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

def dot(T1, T2):
    if T1.dim() == 2:
        T1 = torch.unsqueeze(T1, 0)

    if T2.dim() == 2:
        T2 = torch.unsqueeze(T2, 0)

    R1 = T1[:, :, :3]
    t1 = T1[:, :, 3:4]

    R2 = T2[:, :, :3]
    t2 = T2[:, :, 3:4]

    R = torch.bmm(R1, R2)
    t = torch.bmm(R1, t2) + t1

    return torch.cat([R, t], dim=2)

def inv(T):
    if T.dim() == 2:
        T = T.unsqueeze(dim=0)
    R = T[:, :, :3]
    t = T[:, :, 3:4]

    R_inv = R.transpose(2, 1)
    t_inv = torch.bmm(R_inv, t)
    t_inv = -1. * t_inv
    return torch.cat([R_inv, t_inv], dim=2)


def generate_mean_mesh(K, img_size, scale=0):
    scale = 2 ** scale
    K = K / scale
    img_size = int(img_size // scale)

    _, faces = meshzoo.rectangle(
        xmin = -1., xmax = 1.,
        ymin = -1., ymax = 1.,
        nx = img_size, ny = img_size,
        zigzag=True)

    x = torch.arange(0, img_size, 1).float().cuda() 
    y = torch.arange(0, img_size, 1).float().cuda()
    
    x_ = (x - K[2]) / K[0]
    y_ = (y - K[3]) / K[1]
    
    xx = x_.repeat(img_size, 1)
    yy = y_.view(img_size, 1).repeat(1, img_size)
    zz = torch.ones_like(xx)

    unit_ray = torch.stack([xx, yy, zz], dim=-1) 
    unit_ray = unit_ray.view(-1, 3)
        
    return unit_ray, faces

class warp_image(nn.Module):
    def forward(self, T_cur2ref, K_ref, K_cur, depth_cur, img_ref):
        #inputs are in Batch X original dimension. except K_ref, K_cur
        # K is in format [fx fy cx cy]
        #T_cur2ref = SE3.dot(T_wld2ref, SE3.inv(T_wld2cur))

        [B, C, H, W] = img_ref.size()
        x = torch.arange(0, W, 1).float().cuda()
        y = torch.arange(0, H, 1).float().cuda()
    
        x_ = (x - K_cur[2]) / K_cur[0]
        y_ = (y - K_cur[3]) / K_cur[1]
    
        xx = x_.repeat(H, 1)
        yy = y_.view(H, 1).repeat(1, W)
    
        xxx = xx[None, :, :] * depth_cur
        yyy = yy[None, :, :] * depth_cur
        zzz = depth_cur
     
        p3d_cur = torch.stack([xxx, yyy, zzz], dim=-1) 
        Ones = torch.ones_like(p3d_cur[:, :, :, 0]).unsqueeze(dim=-1)
        p3d_cur = torch.cat([p3d_cur, Ones], dim=-1)
        p3d_cur = p3d_cur.view(B, -1, 4)
        p3d_cur = p3d_cur.transpose(2, 1)

        p3d_ref = T_cur2ref.bmm(p3d_cur)
        p3d_ref = p3d_ref.transpose(2, 1)

        z = p3d_ref[:, :, 2].unsqueeze(dim=-1)
        xy1 = p3d_ref / z 

        K = torch.tensor([[K_ref[0], 0, K_ref[2]], [0, K_ref[1], K_ref[3]], [0, 0, 1]]).float()
        K = K.transpose(1, 0)
        K = K.expand(xy1.shape[0], -1, -1)
        xy1 = torch.bmm(xy1, K)

        # normalize to [-1, 1]
        X = 2.0 * (xy1[:, :, 0] - W * 0.5 + 0.5) / (W - 1.)
        Y = 2.0 * (xy1[:, :, 1] - H * 0.5 + 0.5) / (H - 1.)    
    
        X = X.view(B, H, W)
        Y = Y.view(B, H, W)
        xy = torch.stack([X, Y], dim=-1)

        sample_image = torch.nn.functional.grid_sample(img_ref, xy, padding_mode='zeros') 
        return sample_image 

class SimpleSpline:
    def get_pose(self, t, rel_se3_control_knots):
        da = t * rel_se3_control_knots
        A = SE3.se3_exp(da)
        return A

def generate_blur_image(self, motion_pred, mesh_vertex, mesh_faces, ref_image):
        warped_images = None
        rendered_depth_0 = None

        blur_kernel_half_size = self.opts.blur_kernel_size_max // 2

        for i in range(blur_kernel_half_size + 1):
            # get interpolated current camera pose
            u = i / float(blur_kernel_half_size + 1e-5)

            # world frame is defined as the middle camera frame
            T_wld2last = self.spline.get_pose(u, motion_pred)

            #print('T_wld2last ', T_wld2last)
        
            # render depth for forward motion image
            depth_last = self.renderer(T_wld2last, mesh_vertex, mesh_faces)
            img_last = self.warp_image_fn(self.T_eye, T_wld2last, self.K, self.K, depth_last, ref_image)

            if warped_images is None:
                warped_images = img_last.unsqueeze(-1)
            else:
                warped_images = torch.cat([warped_images, img_last.unsqueeze(-1)], -1)

            if i > 0:
                T_wld2front = self.spline.get_pose(u, -1. * motion_pred)
                depth_front = self.renderer(T_wld2front, mesh_vertex, mesh_faces)
                img_front = self.warp_image_fn(self.T_eye, T_wld2front, self.K, self.K, depth_front, ref_image)
                warped_images = torch.cat([warped_images, img_front.unsqueeze(-1)], -1)   

                #print('T_wld2front ', T_wld2front)       

            if i == blur_kernel_half_size:
                rendered_depth = depth_front.data

        # normalize blur image
        blur_image=torch.mean(warped_images, -1)
        return blur_image, rendered_depth


# create renderer
self.renderer = nr.TorchNeuralRenderer()
self.renderer.initialize(opts.final_img_W, 'depth', self.K)

self.warp_image_fn = image_proc.warp_image()
self.spline = splines.SimpleSpline()

# initialize mesh
vertex_unit_ray_camFrame, faces = geometry.generate_mean_mesh(self.K, opts.final_img_W, scale=self.opts.depth_scale)

if len(opts.gpu_ids) > 0:
    vertex_unit_ray_camFrame = vertex_unit_ray_camFrame.float().cuda()
    faces = torch.from_numpy(faces.astype(np.int32)).cuda()
            
self.mesh_vertex_unit_ray = vertex_unit_ray_camFrame.repeat(self.opts.batch_size, 1, 1)
self.mesh_faces = faces.repeat(self.opts.batch_size, 1, 1)

# convert inverse depth map to mesh
pred_depth = self.pred_depth.view(self.pred_depth.shape[0], -1, 1)
mesh_vertex = self.mesh_vertex_unit_ray * pred_depth

## compute self consistency loss
img_synth, rendered_depth = self.generate_blur_image(pred_motion, mesh_vertex, self.mesh_faces, target_sharp_image)