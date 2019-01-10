# Johann Diep (jdiep@student.ethz.ch) - January 2019

# This file sets up the renderer.

# libraries
from lib import dataset

# external libraries
import torch
import torch.nn as nn
import neural_renderer as nr


# defining the renderer
class Renderer(dataset.Intrinsics, nn.Module):
    def __init__(self, cam_index, vertices, faces):
        super(Renderer, self).__init__()

        self.texture_size = 2 # texture parameter, not relevant

        # initializing buffers, only vertices and faces are important, dont care about texture
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])
        textures = torch.ones(1, self.faces.shape[1], self.texture_size, self.texture_size, self.texture_size, 3,
                              dtype = torch.float32).cuda()
        self.register_buffer('textures', textures)

        # initialzing ProjectiveRenderer-object
        img_size_x, img_size_y, K = self.get_intrinsics(cam_index)
        K = torch.tensor(K).cuda().float()
        renderer = nr.ProjectiveRenderer(image_size = img_size_x, K = torch.unsqueeze(K, 0).cuda())
        self.renderer = renderer