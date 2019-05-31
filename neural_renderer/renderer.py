from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.optim as optim
from torch.utils.dlpack import to_dlpack, from_dlpack
import numpy as np
import neural_renderer as nr
from neural_renderer import utils

PI                                  = 3.1415926535897932384626

class Renderer(nn.Module):
    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0,0,0],
                 fill_back=True, camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=1024,
                 perspective=True, viewing_angle=30, camera_direction=[0,0,1],
                 near=0.1, far=100,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,1,0]):
        super(Renderer, self).__init__()
        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        # camera
        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.P = P
            if isinstance(self.P, np.ndarray):
                self.P = torch.from_numpy(self.P).cuda()
            if self.P is None or P.ndimension() != 3 or self.P.shape[1] != 3 or self.P.shape[2] != 4:
                raise ValueError('You need to provide a valid (batch_size)x3x4 projection matrix')
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(P.shape[0], 1)
            self.orig_size = orig_size
        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
            self.camera_direction = [0, 0, 1]
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')


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

    def forward(self, vertices, faces, textures=None, mode=None):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''
        
        if mode is None:
            return self.render(vertices, faces, textures)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces)
        elif mode == 'depth':
            return self.render_depth(vertices, faces)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render(self, vertices, faces, textures):
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

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images


class UVNeuralRenderer(nn.Module):
    def __init__(self,img_size=256, cuda=True, fill_back=False,
               camera_mode='projection', camera_direction=[0, 0, 0],
               camera_up=[0, 1, 0], near=0.1, far=100, perspective=True,
               bkg_colour=[0,0,0], anti_aliasing=True,
               viewing_angle=30,eps=1e-3):
        super(UVNeuralRenderer, self).__init__()
        self.img_size = img_size
        self.cuda = cuda
        self.fill_back = fill_back

        # camera
        self.perspective = perspective
        self.camera_mode = camera_mode
        self.camera_direction = camera_direction
        self.camera_up = camera_up
        self.near = near
        self.far = far
        self.bkg_colour = bkg_colour
        self.eps = eps
        self.fill_back = fill_back
        self.anti_aliasing = anti_aliasing
        self.viewing_angle = viewing_angle
        self.eye = Variable(torch.FloatTensor(
            [0, 0, -(1. / np.tan(self.viewing_angle*PI/180.0)
                + 1)]).cuda())

        # Rasteriser 
        self.rasteriser = nr.Rasterize(self.img_size, self.near,
                self.far, self.eps, self.bkg_colour, return_depth=True,
                return_rgb=True, return_alpha=True)

    def transform_vertices(self, vertices, camera_params, force_camera=None):
        # Transform according to camera. 
        if force_camera is not None:
            ceye = utils.get_points_from_angles(force_camera[:,0],
                    force_camera[:,1], force_camera[:,2], degrees=False)
            cat = force_camera[:,3:6]
            cup = force_camera[:,6:9]
            vertices = utils.look_at(vertices, ceye, at=cat, up=cup)
        else:
            fixed_cam_params = utils.fix_cam_params(camera_params)

            vertices = utils.orthographic_proj_withz(
                    vertices, fixed_cam_params, offset_z=5)
            eye = utils.get_points_from_angles(2.732, 0, 0, degrees=False)
            vertices = utils.look_at(
                    vertices, eye, at=self.camera_direction, up=self.camera_up)

        # Apply perspective projection if necessary 
        if self.perspective:
            vertices = utils.perspective(vertices, angle=self.viewing_angle)
        return vertices
    
    def forward(self, uv, in_faces,
            vertices, camera_params=None, force_camera=None):
        # fill back
        if self.fill_back:
            faces_inv_idx = Variable(torch.arange(
                in_faces.size(2)-1, -1, -1).long().cuda())
            faces_inv = torch.index_select(in_faces, 2, faces_inv_idx)
            faces = torch.cat((in_faces, faces_inv), dim=1)
        else:
            faces = in_faces

        # Transform vertices according to the projection. 
        # Also includes perspective transform, if specified. 
        vertices = self.transform_vertices(
                vertices, camera_params, force_camera=force_camera)
        self.projected_vertices = vertices    
        faces_uv = utils.vertices_to_faces(uv, faces)
        faces = utils.vertices_to_faces(vertices, faces)
        return self.rasteriser(faces, faces_uv)
