'''
@JingwenWang95's code for preprocessing the data. This code is 
used to generate the virtual cameras & cropped frames. See the
supplementary material for more details and explanations.
'''

import os
import sys
sys.path.append('..')
import cv2
import copy
import yaml
import torch
import imageio
import argparse
import numpy as np

from glob import glob
from utils import safe_normalize, load_K_Rt_from_P, gl2cv

class Database():
    def __init__(self, config, align=True):
        self.cfg = config
        self.align = align
        self.data_dir = config['data']['data_dir']
        self.render_cameras_name = config['data']['render_cameras_name']
        self.object_cameras_name = config['data']['object_cameras_name']
        
        self.size_h = self.cfg['data']['size_h']
        self.size_w = self.cfg['data']['size_w']
        self.rot_degree = self.cfg['data']['rot_degree']
        
        self.align_mat = self.get_align_mat()
        
        
        
        self.images = self.get_rgbs(self.data_dir)
        self.n_images = len(self.images)
        
        self.depths = self.get_depths(self.data_dir)
        self.masks = self.get_masks(self.data_dir)
        self.poses, self.intrinsics, self.scales, self.scale_mats = self.get_camera_params(self.data_dir)
        self.scale_depth()
        
        
        
        self.H, self.W = self.depths.shape[1:3]
    
    def scale_depth(self):
        if len(self.scales.shape) != len(self.depths.shape):
            scales = self.scales.unsqueeze(-1)
        else:
            scales = self.scales
        self.depths *= scales.numpy()
    
    def get_align_mat(self):
        align_mat = torch.eye(4)
        align_mat[1, 1] = -1.
        align_mat[2, 2] = -1.
        return align_mat
        
    def get_rgbs(self, data_dir):
        print('loading rgbs')
        images_lis = sorted(glob(os.path.join(data_dir, 'rgb/*.jpg')))
        if len(images_lis) == 0:
            images_lis = sorted(glob(os.path.join(data_dir, 'rgb/*.png')))
        return np.stack([cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB) for im_name in images_lis]) / 255.0
    
    def get_depths(self, data_dir):
        print('loading depths')
        depth_scale = self.cfg['data']['depth_scale']
        depths_lis = sorted(glob(os.path.join(data_dir, 'depth/*.jpg')))
        if len(depths_lis) == 0:
            depths_lis = sorted(glob(os.path.join(data_dir, 'depth/*.png')))[:self.n_images]
        depths_np = np.stack([cv2.imread(im_name, cv2.IMREAD_UNCHANGED) for im_name in depths_lis]) / depth_scale
        depths = depths_np.astype(np.float32)
        return depths
    
    def get_masks(self, data_dir):
        print('loading masks')
        masks_lis = sorted(glob(os.path.join(data_dir, 'mask/*.jpg')))
        if len(masks_lis) == 0:
            masks_lis = sorted(glob(os.path.join(data_dir, 'mask/*.png')))[:self.n_images]
        masks_np = np.stack([cv2.imread(im_name)[...,0] for im_name in masks_lis]) / 255.0

        return masks_np
    
    def get_camera_params(self, data_dir):
        camera_dict = np.load(os.path.join(data_dir, self.render_cameras_name))
        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        intrinsics_all = []
        poses_all = []
        # Depth, needs x,y,z have equal scale
        scales_all = []

        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            # Depth
            scales_all.append(torch.from_numpy(np.array([1.0/scale_mat[0,0]])))
            intrinsics_all.append(torch.from_numpy(intrinsics))
            pose = torch.from_numpy(pose)
            # opencv to opengl
            pose[:3, 1] *= -1
            pose[:3, 2] *= -1
            if self.align:
                pose_align = self.align_mat @ pose
            else:
                pose_align = pose
            poses_all.append(pose_align)  # the inverse of extrinsic matrix

        scales_all = torch.stack(scales_all)
        intrinsics_all = torch.stack(intrinsics_all)
        poses_all = torch.stack(poses_all)

        return poses_all, intrinsics_all, scales_all, scale_mats_np
    
    def get_frame(self, idx):
        ret = {
            "frame_id": torch.tensor(idx),
            "c2w": self.poses[idx],
            "rgb": self.images[idx],
            "depth": self.depths[idx],
            "mask": self.masks[idx],
            "time": torch.tensor(idx)/self.n_images,
            "K": self.intrinsics[idx]     
            }
        return ret
        


class DataProcessor(Database):
    def __init__(self, config, align=True):
        super().__init__(config, align)
    
    
    def save_intrinsics(self):
        intrinsic_path = os.path.join(self.data_dir, file_name)
        
        if os.path.exists(intrinsic_path):
            print("Intrinsics file already exists")    
        else:
            np.savetxt(intrinsic_path, self.intrinsics[0])
    
    def save_polar(self):        
        rtp = torch.stack([self.radius, self.theta, self.phi], dim=-1)  # [B, 3]
        raw_rtp = torch.stack([self.raw_radius, self.raw_theta, self.raw_phi], dim=-1)
        np.savetxt(os.path.join(self.data_dir, "r_theta_phi.txt"), rtp)
        np.savetxt(os.path.join(self.data_dir, "raw_r_theta_phi.txt"), raw_rtp)
    
    def get_x_vector_from_c2w(self, c2ws):
        x_vectors = []
        for c2w in c2ws:
            x_vectors.append(c2w[:3, 0])
        
        return torch.stack(x_vectors, dim=0)
    
    def get_c2w_from_cam_center(self, cam_centers, targets=0, x_axis=None, keep_chirality=True, camera_convention=None):
        """
        Calculates the camera-to-world transformation matrix based on the camera centers and targets.

        """
        if x_axis is None:
            if camera_convention == "OpenGL":
                forward_vector = safe_normalize(cam_centers - targets)
                up_vector = torch.FloatTensor([0, 1, 0]).unsqueeze(0).repeat(forward_vector.shape[0], 1)

            elif camera_convention == "OpenCV":
                forward_vector = safe_normalize(targets - cam_centers)
                up_vector = torch.FloatTensor([0, -1, 0]).unsqueeze(0).repeat(bs, 1)

            else:
                raise NotImplementedError

            if keep_chirality:
                right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))
                up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1))
            else:
                right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
                up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))
        else:
            if camera_convention == "OpenGL":
                forward_vector = safe_normalize(cam_centers - targets)
            elif camera_convention == "OpenCV":
                forward_vector = safe_normalize(targets - cam_centers)
            else:
                raise NotImplementedError

            right_vector = x_axis
            if keep_chirality:
                up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1))
            else:
                up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

        poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
        poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
        poses[:, :3, 3] = cam_centers
        
        return poses
    
    def get_c2w_from_polar(self,
                           t=None,
                           keep_chirality=True,
                           x_axis=None,  # [bs, 3]
                           camera_convention="OpenGL",
                           radius=None,
                           theta=torch.tensor([60.0]),
                           phi=torch.tensor([0.0]),
                           angle_overhead=30,
                           angle_front=60,
                           radius_scale=None):
        """
        Converts polar coordinates to camera-to-world transformation matrices.

        Args:
            keep_chirality (bool): Whether to keep the chirality of the camera coordinate system.
            x_axis (torch.Tensor): The x-axis vector of the camera coordinate system.
            camera_convention (str): The camera convention to use. Possible values are "OpenGL" and "OpenCV".
            radius (torch.Tensor): The radius of the polar coordinates.
            theta (torch.Tensor): The theta angle of the polar coordinates.
            phi (torch.Tensor): The phi angle of the polar coordinates.
            return_dirs (bool): Whether to return the view directions.
            angle_overhead (float): The overhead angle in degrees.
            angle_front (float): The front angle in degrees.

        Returns:
            torch.Tensor: The camera-to-world transformation matrices.
            torch.Tensor: The view directions (if return_dirs is True).

        """
        if t is not None:
            radius = self.get_radius(t)
        
        if radius_scale is not None:
            radius = radius_scale * radius
        
            
        theta = theta / 180 * np.pi
        phi = phi / 180 * np.pi
        angle_overhead = angle_overhead / 180 * np.pi
        angle_front = angle_front / 180 * np.pi
        bs = theta.shape[0]

        cam_centers = torch.stack([
            radius * torch.sin(theta) * torch.sin(phi),
            radius * torch.cos(theta),
            radius * torch.sin(theta) * torch.cos(phi),
        ], dim=-1)  # [B, 3]

        targets = 0

        poses = self.get_c2w_from_cam_center(cam_centers, targets, 
                                             x_axis=x_axis, 
                                             keep_chirality=keep_chirality, 
                                             camera_convention=camera_convention)

        return poses
                
    def get_polar_from_c2w(self, c2ws, virtual=False, scale_radius=1.0):
        # Get the polar coordinates from the camera-to-world matrix
        
        radius, thetas, phis = [], [], []
        
        for c2w in c2ws:
            cam_centre = c2w[:3, 3]
            cam_z_dir = c2w[:3, 2]
            
            
            if virtual:
                r = (cam_centre * cam_z_dir).sum()
                theta = torch.acos(cam_z_dir[1])
                phi = torch.atan2(cam_z_dir[0], cam_z_dir[2])
            else:
                r = torch.sqrt(torch.sum(cam_centre ** 2))
                cam_centre_uni = cam_centre / r
                theta = torch.acos(cam_centre_uni[1])
                phi = torch.atan2(cam_centre_uni[0], cam_centre_uni[2])
            
            if phi < 0.:
                phi += 2 * np.pi
            
            theta = theta * 180. / np.pi
            phi = phi * 180. / np.pi
            
            radius.append(r * scale_radius)
            thetas.append(theta)
            phis.append(phi)
        
        return torch.tensor(radius), torch.tensor(thetas), torch.tensor(phis)
                  
    def get_virtual_views(self, scale_radius=1.0):
        self.radius, self.theta, self.phi = self.get_polar_from_c2w(self.poses, virtual=True, scale_radius=scale_radius)
        self.raw_radius, self.raw_theta, self.raw_phi = self.get_polar_from_c2w(self.poses, virtual=False, scale_radius=scale_radius)
        
        x_vectors = self.get_x_vector_from_c2w(self.poses)
        self.poses_virt = self.get_c2w_from_polar(radius=self.radius, x_axis=x_vectors, theta=self.theta, phi=self.phi)
        
        fx, fy = self.intrinsics[0, 0, 0], self.intrinsics[0, 1, 1]
        H, W = self.size_h, self.size_w
        
        self.intrinsics_virt = np.array([[fx, 0., W/2],
                                             [0., fy, H/2],
                                             [0., 0., 1.]])
        
    def crop_image_2d(self, img, top, left, h, w, return_mask=False):
        crop = np.zeros((h, w))
        if top >= 0 and left >= 0 and top + h < self.H and left + w < self.W:
            crop = img[top:top+h, left:left+w]
            padding_mask = np.zeros((h, w))
        elif top < 0 or left < 0:  # out of lower bound
            padding_mask = np.ones((h, w))
            if top < 0 and left < 0:
                offset_h, offset_w = -top, -left
                crop[offset_h:, offset_w:] = img[:top+h, :left+w]
                padding_mask[offset_h:, offset_w:] = 0.
            elif top < 0:
                offset_h = -top
                crop[offset_h:, :] = img[:top+h, left:left+w]
                padding_mask[offset_h:, :] = 0.
            else:
                offset_w = -left
                crop[:, offset_w:] = img[top:top+h, :left+w]
                padding_mask[:, offset_w:] = 0.
        else:  # out of upper bound
            padding_mask = np.ones((h, w))
            if top + h >= self.H and left + w >= self.W:
                dh = self.H - top
                dw = self.W - left
                crop[:dh, :dw] = img[top:, left:]
                padding_mask[:dh, :dw] = 0.
            elif top + h >= self.H:
                dh = self.H - top
                crop[:dh, :] = img[top:, left:left+w]
                padding_mask[:dh, :] = 0.
            else:
                dw = self.W - left
                crop[:, :dw] = img[top:top+h, left:]
                padding_mask[:, :dw] = 0.

        if return_mask:
            return crop, padding_mask
        else:
            return crop
    
    def crop_image_3d(self, img, top, left, h, w):
        crop = np.zeros((h, w, img.shape[-1]))
        if top >= 0 and left >= 0 and top + h < self.H and left + w < self.W:
            crop = img[top:top+h, left:left+w, :]
        elif top < 0 or left < 0:  # out of lower bound
            if top < 0 and left < 0:
                offset_h, offset_w = -top, -left
                crop[offset_h:, offset_w:, :] = img[:top+h, :left+w, :]
            elif top < 0:
                offset_h = -top
                crop[offset_h:, :, :] = img[:top+h, left:left+w, :]
            else:
                offset_w = -left
                crop[:, offset_w:, :] = img[top:top+h, :left+w, :]
        else:  # out of upper bound
            if top + h >= self.H and left + w >= self.W:
                dh = self.H - top
                dw = self.W - left
                crop[:dh, :dw, :] = img[top:, left:, :]
            elif top + h >= self.H:
                dh = self.H - top
                crop[:dh, :, :] = img[top:, left:left+w, :]
            else:
                dw = self.W - left
                crop[:, :dw, :] = img[top:top+h, left:, :]
        return crop
    
    def set_crop_mask(self, mask, top, left, h, w):
        if top >= 0 and left >= 0 and top + h < self.H and left + w < self.W:
            mask[top:top+h, left:left+w] = 1.0
        elif top < 0 or left < 0:  # out of lower bound
            if top < 0 and left < 0:
                mask[:top+h, :left+w] = 1.0
            elif top < 0:
                mask[:top+h, left:left+w] = 1.0
            else:
                mask[top:top+h, :left+w] = 1.0
        else:  # out of upper bound
            if top + h >= self.H and left + w >= self.W:
                mask[top:, left:] = 1.0
            elif top + h >= self.H:
                mask[top:, left:left+w] = 1.0
            else:
                mask[top:top+h, left:] = 1.0
    
    def rotate_and_crop_frames(self):
        # Real pose here
        rgb_rot, rgb_rot_crop, depth_rot, depth_rot_crop, mask_rot, mask_rot_crop, c2w_list = [], [], [], [], [], [], []
        
        frame_ids = []
        rgb_virt = []
        depth_raw_crop = []
        mask_virt = []
        padding_masks = []
        world_centre = np.zeros((3,))
        crop_centres = []
        center_rot = []
        
        for i in range(self.n_images):
            data = self.get_frame(i)
            c2w, rgb, depth, mask = data["c2w"], data["rgb"], data["depth"], data["mask"]
            K = data["K"][:3, :3].numpy()
            c2w = gl2cv(c2w)
            w2c = np.linalg.inv(c2w)
            c2w_list.append(c2w)
            x_c = w2c[:3, :3] @ world_centre + w2c[:3, -1]
            p_xyz = K @ x_c
            px, py, pz = p_xyz[0], p_xyz[1], p_xyz[2]
            px = int(px / pz)
            py = int(py / pz)

            center = (px, py)
            
            center_rot.append(center)
            crop_centres.append(np.array(center))
            
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rot_degree, 1.0)
            
            # Perform the rotation on the image
            rgb = cv2.warpAffine(rgb, rotation_matrix, (self.W, self.H))
            depth = cv2.warpAffine(depth, rotation_matrix, (self.W, self.H), flags=cv2.INTER_NEAREST)
            mask = cv2.warpAffine(mask, rotation_matrix, (self.W, self.H), flags=cv2.INTER_NEAREST)
            rgb_rot.append(rgb)
            depth_rot.append(depth)
            mask_rot.append(mask)
            
            # crop (not real crop, but setting pixels outside to zero)
            crop_mask = np.zeros_like(depth)
            top, left = py - self.size_h // 2 + 1, px - self.size_w // 2 + 1
            # crop_mask[top:top+self.size_h, left:left+self.size_w] = 1.0
            self.set_crop_mask(crop_mask, top, left, self.size_h, self.size_w)
            rgb_crop = copy.deepcopy(rgb)
            rgb_crop[crop_mask <= 0.0, :] = 0.0
            depth_crop = copy.deepcopy(depth)
            depth_crop[crop_mask <= 0.0] = 0.0
            mask_crop = copy.deepcopy(mask)
            mask_crop[crop_mask <= 0.0] = 0.0
            rgb_rot_crop.append(rgb_crop)
            depth_rot_crop.append(depth_crop)
            mask_rot_crop.append(mask_crop)
            
            
            rgb_virt.append(self.crop_image_3d(rgb, top, left, self.size_h, self.size_w))
            depth_raw_crop.append(self.crop_image_2d(depth, top, left, self.size_h, self.size_w))
            mask_v, padding_mask = self.crop_image_2d(mask, top, left, self.size_h, self.size_w, return_mask=True)
            mask_virt.append(mask_v)
            padding_masks.append(padding_mask)
            frame_ids.append(i)
        
        np.savetxt(os.path.join(self.data_dir, "crop_centre_list.txt"), np.stack(crop_centres, 0))
        
        data = {
            "rgb": rgb_rot,
            "rgb_crop": rgb_rot_crop,
            "rgb_virt": rgb_virt,
            "depth": depth_rot,
            "depth_crop": depth_rot_crop,
            "depth_raw_crop": depth_raw_crop, 
            "mask": mask_rot,
            "mask_crop": mask_rot_crop,
            "mask_virt": mask_virt,
            "padding_masks": padding_masks,
            "c2w": c2w_list,
            "K": K,
            "frame_ids": frame_ids
        }

        return data
            
    def preprocess(self):
        # self.save_intrinsics()
        self.get_virtual_views(scale_radius=1.0)
        self.save_polar()
        
        data = self.rotate_and_crop_frames()
        
        rgb_dir = os.path.join(self.data_dir, "color_virt")
        depth_dir = os.path.join(self.data_dir, "depth_raw_crop")
        mask_dir = os.path.join(self.data_dir, "mask_virt")
        pose_dir = os.path.join(self.data_dir, "poses_virt")
        padding_mask_dir = os.path.join(self.data_dir, "padding_mask")
        
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)
        os.makedirs(padding_mask_dir, exist_ok=True)
        
        np.savetxt(os.path.join(self.data_dir, "K_virt.txt"), self.intrinsics_virt)
                
        for i in range(self.n_images):
            # Camera pose
            c2w_virt = self.poses_virt[i]
            np.savetxt(os.path.join(pose_dir, "{:06d}.txt".format(i)), c2w_virt.cpu().numpy())  # [B, 4, 4]
            
            # Images
            rgb_virt =data['rgb_virt'][i]
            d_virt = data['depth_raw_crop'][i]
            mask_virt = data['mask_virt'][i]
            padding_mask = data['padding_masks'][i]
            
            imageio.imwrite(os.path.join(depth_dir, "{:06d}.png".format(i)), (d_virt * 1000).astype(np.uint16))
            imageio.imwrite(os.path.join(rgb_dir, "{:06d}.png".format(i)), (rgb_virt * 255.).astype(np.uint8))
            imageio.imwrite(os.path.join(mask_dir, "{:06d}.png".format(i)), (mask_virt * 255.).astype(np.uint8))
            imageio.imwrite(os.path.join(padding_mask_dir, "{:06d}.png".format(i)), (padding_mask * 255.).astype(np.uint8))
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")
    parser.add_argument('--config', type=str, default='configs/snoopy.yaml', help='Path to the YAML config file')
    
    args = parser.parse_args()
    config = yaml.full_load(open(args.config, 'r'))
    
    dp = DataProcessor(config, align=True)
    dp.preprocess()
    
        