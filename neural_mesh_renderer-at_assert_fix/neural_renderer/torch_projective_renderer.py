from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.misc
import tqdm
import torch
import neural_renderer as nr

########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class TorchNeuralRenderer(torch.nn.Module):
    def __init__(self):
        super(TorchNeuralRenderer, self).__init__()
        
    def initialize(self, image_size, mode, K, dist_coeffs=None):
        self.mode = mode
        self.image_size = image_size
        self.K = K

    def forward(self, T, vertices, faces, textures=None):
        if textures is None:
            return nr.ProjectiveRenderer(image_size=self.image_size, K=self.K)(T, vertices, faces, mode=self.mode)
        else:
            return nr.ProjectiveRenderer(image_size=self.image_size, K=self.K)(T, vertices, faces, textures, mode=self.mode)

########################################################################
############################## Tests ###################################
########################################################################
