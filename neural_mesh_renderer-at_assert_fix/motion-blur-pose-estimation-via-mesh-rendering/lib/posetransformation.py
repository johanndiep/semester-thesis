# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file implement the used mapping between Lie group SE(3) and Lie algebra se(3).


# external libraries
import torch


# pose calculation from Lie group to homogeneous transformation
class PoseTransformation():
    def __init__(self):
        super(PoseTransformation, self).__init__()

    # inplement the exponential se(3)-mapping    
    def se3_exp(self, tangent):
        # set to dimension 2
        if tangent.dim() < 2:
            tangent = tangent.unsqueeze(dim = 0)

        # extract se(3)-translation and se(3)-rotation
        t = tangent[:, :3]
        phi = tangent[:, 3:]

        R, R_jac = self.so3_exp(phi) # calculating SO(3)-rotation matrix and V
        t = t.unsqueeze(dim = 2) # set to the right dimension

        trans = torch.bmm(R_jac, t) # calculating SE(3)-translation
        return torch.cat([R, trans], dim = 2) # return SO(3)-pose

    # implement the exponential so(3)-mapping    
    def so3_exp(self, phi):
        # set to dimension 2
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim = 0)

        # V matrix size
        jac = phi.__class__(phi.shape[0], 3, 3)
        I = torch.eye(3).expand_as(jac)

        # norm of se(3)-rotation
        angle = phi.norm(p = 2, dim = 1)
        angle = angle + 1e-5

        # calculating the taylor expansions
        s = angle.sin()
        c = angle.cos()
        s_div_angle = s / angle
        one_minus_s_div_angle = 1. - s_div_angle
        one_minus_c = 1. - c
        one_minus_c_div_angle = one_minus_c / angle
        
        # set to the right dimension
        s_div_angle = s_div_angle.unsqueeze(1).unsqueeze(2)
        one_minus_s_div_angle = one_minus_s_div_angle.unsqueeze(1).unsqueeze(2)
        one_minus_c_div_angle = one_minus_c_div_angle.unsqueeze(1).unsqueeze(2)
        c = c.unsqueeze(1).unsqueeze(2)
        s = s.unsqueeze(1).unsqueeze(2)
        one_minus_c = one_minus_c.unsqueeze(1).unsqueeze(2)

        # get unit axis of rotation
        angle = angle.unsqueeze(1)
        axis = phi / angle

        axis2 = self.outer(axis, axis) # calculate squared skew-matrix 
        wedge_axis = self.wedge(axis) # calculate skew_matrix

        # calculating the taylor expansions
        A_jac = s_div_angle * I
        B_jac = one_minus_s_div_angle * axis2
        C_jac = one_minus_c_div_angle * wedge_axis
        A = c * I
        B = one_minus_c * axis2
        C = s * wedge_axis

        return A + B + C, A_jac + B_jac + C_jac # return the SO(3)-rotation and V-matrix

    # construct squared skew-matrix
    def outer(self, vecs1, vecs2):
        # set to dimension 2
        if vecs1.dim() < 2:
            vecs1 = vecs1.unsqueeze(dim = 0)
        if vecs2.dim() < 2:
            vecs2 = vecs2.unsqueeze(dim = 0)

        # error message if size is wrong
        if vecs1.shape[0] != vecs2.shape[0]:
            raise ValueError("Inconsistent batch sizes {} and {}".format(vecs1.shape[0], vec2.shape[0]))

        # return squared skew-matrix
        return torch.bmm(vecs1.unsqueeze(dim = 2), vecs2.unsqueeze(dim = 2).transpose(2, 1))[0,:,:]

    # construct skew-matrix    
    def wedge(self, phi):
        # set to dimension 2
        if phi.dim() < 2:
            phi.unsqueeze(dim = 0)

        # error message if size is wrong
        if phi.shape[1] != 3:
            raise ValueError("phi must have shape ({},) or (N,{})".format(3, 3))

        # return skew-matrix 
        Phi = phi.__class__(phi.shape[0], 3, 3)
        Phi[0, :, :] = torch.zeros(3, 3)
        Phi[:, 0, 1] = -phi [:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]

        return Phi[0, :, :] # reduce dimension

    # transforms a translation from SE(3) to se(3)
    def from_SE3t_to_se3u(self, q, t):
        # reading input in the right form
        quat = q
        t = t.unsqueeze(dim = 0).unsqueeze(dim = 1).cuda().float()

        # norm and rotation-axis of se(3)-rotation 
        axis = torch.tensor(quat.axis).cuda().float().unsqueeze(dim = 0)
        angle = torch.tensor([quat.angle]).cuda().float().unsqueeze(dim = 0)

        # calculate the taylor expansions
        A = angle.sin() / angle
        B = (1 - angle.cos()) / (angle * angle)
        C = 1 / (angle * angle) *  (1 - A / (2 * B))

        # calculating the inverse V
        I = torch.eye(3).cuda().unsqueeze(dim = 0)
        V_inverse = I - 0.5 * self.wedge(axis * angle) + C * torch.bmm(self.wedge(axis * angle).unsqueeze(dim = 0), self.wedge(axis * angle).unsqueeze(dim = 0))

        return torch.bmm(V_inverse, t.transpose(1,2))[0,:,0] # return se(3)-translation

    # transforms a rotation in SE(3) to se(3)
    def from_SE3rot_to_se3w(self, r):
        # reading and storing rotional elements
        m_00 = r[:, 0, 0]
        m_11 = r[:, 1, 1]
        m_22 = r[:, 2, 2]
        m_01 = r[:, 0, 1]
        m_02 = r[:, 0, 2]
        m_10 = r[:, 1, 0]
        m_12 = r[:, 1, 2]
        m_20 = r[:, 2, 0]
        m_21 = r[:, 2, 1]

        # transforming rotational matrix to angle axis form
        angle = ((m_00 + m_11 + m_22 -1) / 2).acos()
        ax = (m_21 - m_12) / ((m_21 - m_12).pow(2) + (m_02 - m_20).pow(2) + (m_10 - m_01).pow(2)).sqrt()
        ay = (m_02 - m_20) / ((m_21 - m_12).pow(2) + (m_02 - m_20).pow(2) + (m_10 - m_01).pow(2)).sqrt() 
        az = (m_10 - m_01) / ((m_21 - m_12).pow(2) + (m_02 - m_20).pow(2) + (m_10 - m_01).pow(2)).sqrt()

        return torch.cat([ax * angle, ay * angle, az * angle], dim = 0).unsqueeze(dim = 0) # return angle axis