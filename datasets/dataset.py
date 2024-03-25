import os
import cv2
import copy
import torch
import random
import numpy as np
import torch.nn.functional as F

from glob import glob
from torch.utils.data import Dataset
from .utils import get_camera_rays, safe_normalize, get_view_direction, load_K_Rt_from_P


class BaseDataset():
    """
    Base class for a dataset.

    Args:
        config (dict): Configuration parameters for the dataset.
        test_id (list): List of indices to select specific samples for testing. Default is None.
        load (bool): Flag indicating whether to load the dataset. Default is True.
        outlier_remove (bool): Flag indicating whether to remove outliers from the dataset. Default is True.
    """

    def __init__(self, config, test_id=None, load=True, outlier_remove=True):
        self.cfg = config
        self.data_dir = config['data']['data_dir']
        
        if load:
            # (Bs, H, W, 3) ; (Bs, H, W) ; (Bs, H, W)
            self.images, self.depths, self.masks = self.load_data(test_id)
            self.num_frames = self.images.shape[0]
        else:
            self.images, self.depths, self.masks = None, None, None
        
        self.intrinsics = self.load_intrinsic() # (3, 3)
        self.radius, self.theta, self.phi = self.load_r_theta_phi() # (Bs)
        self.poses = self.load_poses() # (Bs, 4, 4)
        self.bounding_box = self.load_bounding_box()
        self.bound = self.bounding_box.abs().max()
        self.H, self.W = self.get_HW()
        
        self.outlier_remove = outlier_remove
      
    def load_data(self, test_id=None):
        """
        Load the dataset.

        Args:
            test_id (list): List of indices to select specific samples for testing. Default is None.
        """
        print('Real depth')
        p_images = sorted(glob(os.path.join(self.data_dir, 'color_virt/*.png')))
        p_depths = sorted(glob(os.path.join(self.data_dir, 'depth_raw_crop/*.png')))
        p_masks = sorted(glob(os.path.join(self.data_dir, 'mask_virt/*.png')))
        
        if test_id is not None:
            p_images = [p_images[i] for i in test_id]
            p_depths = [p_depths[i] for i in test_id]
            p_masks = [p_masks[i] for i in test_id]
        
        images = np.stack([cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB) for im_name in p_images]) / 255.0
        depths = np.stack([cv2.imread(im_name, cv2.IMREAD_UNCHANGED) for im_name in p_depths]) / self.cfg['data']['depth_scale']
        masks = np.stack([cv2.imread(im_name, cv2.IMREAD_UNCHANGED) for im_name in p_masks]) / 255.0
        
        return images, depths, masks
            
    def load_intrinsic(self):
        """
        Load the intrinsic matrix.

        Returns:
            torch.Tensor: The loaded intrinsic matrix.
        """
        return torch.from_numpy(np.loadtxt(os.path.join(self.data_dir, 'K_virt.txt')))
    
    def remove_outlier(self, poses, thresh=2.0):
        """
        Remove outliers from the poses. This function is far
        from perfect, but can remove outliers in duck sequence,
        which is quite difficult to handle.

        Args:
            poses (torch.Tensor): The poses to remove outliers from.
            thresh (float): The threshold for outlier removal. Default is 2.0.

        Returns:
            torch.Tensor: The poses with outliers removed.
        """
        trans = poses[:, :3, 3]
        rots = poses[:, :3, :3]
        
        diff = (trans[1:] - trans[:-1]).square().sum(-1).sqrt()
        
        z_scores = (diff - torch.mean(diff)) / torch.std(diff)
        outlier_indices = torch.where(torch.abs(z_scores) > thresh)[0]
        
        trans_new = copy.deepcopy(trans)
        pose_new = copy.deepcopy(poses)
        
        final_outliers = []
        
        for i in outlier_indices:
            index = i + 1
            
            while True:   
                if index > self.num_frames-1:
                    break
                                 
                prev_diff = (trans_new[index] - trans_new[index-1]).square().sum(-1).sqrt()
                
                prev_z_s = (prev_diff - torch.mean(diff)) / torch.std(diff)
                
                if prev_z_s > thresh:
                    final_outliers.append(copy.deepcopy(index))
                    trans_new[index] = copy.deepcopy(trans_new[index-1])
                    pose_new[index] = copy.deepcopy(pose_new[index-1])
                    self.theta[index] = copy.deepcopy(self.theta[index-1])
                    self.phi[index] = copy.deepcopy(self.phi[index-1])
                    self.radius[index] = copy.deepcopy(self.radius[index-1])
                    
                    if index > self.num_frames-2:
                        break
                    next_trans = trans_new[index+1]
                    
                    
                    next_diff = (next_trans - trans_new[index]).square().sum(-1).sqrt()
                    
                    
                    next_z_s = (next_diff - torch.mean(diff)) / torch.std(diff)
                    
                    
                    if next_z_s > thresh:
                        index += 1
                    else:
                        break
                else:
                    break
        
        print('Outlier removed:', final_outliers)
        
        
        return pose_new
    
    def load_poses(self):
        """
        Load the poses.

        Returns:
            torch.Tensor: The loaded poses.
        """
        p_poses = sorted(glob(os.path.join(self.data_dir, 'poses_virt/*.txt')))
        poses = []
        for p in p_poses:
            c2w = np.loadtxt(p)            
            poses.append(c2w)
            # poses.append(np.loadtxt(p))
        poses_raw = torch.from_numpy(np.stack(poses)).to(torch.float32)
        
        if self.cfg['data']['outlier_remove']:
            return self.remove_outlier(poses_raw)
        else:
            return poses_raw
    
    def load_r_theta_phi(self):
        """
        Load the radius, theta, and phi values.

        Returns:
            tuple: A tuple containing the loaded radius, theta, and phi values.
        """
        r_theta_phi = np.loadtxt(os.path.join(self.data_dir, 'r_theta_phi.txt'))
        
        radius = torch.from_numpy(r_theta_phi[:, 0]).to(torch.float32)
        theta = torch.from_numpy(r_theta_phi[:, 1]).to(torch.float32)
        phi = torch.from_numpy(r_theta_phi[:, 2]).to(torch.float32)
        
        return radius, theta, phi

    def load_bounding_box(self):
        """
        Load the bounding box.

        Returns:
            torch.Tensor: The loaded bounding box.
        """
        return torch.tensor([-1.01, -1.01, -1.01, 1.01, 1.01, 1.01]).to(torch.float32)
    
    def get_HW(self):
        """
        Get the height and width of the dataset.

        Returns:
            tuple: A tuple containing the height and width of the dataset.
        """
        if self.images is not None:
            return self.images.shape[1], self.images.shape[2]
        else:
            p_image = glob(os.path.join(self.data_dir, 'color_virt/*.png'))[0]
            image = cv2.imread(p_image)
            return image.shape[0], image.shape[1]
            
      
class DeformDataset(BaseDataset):
    def __init__(self, config, is_train=True, load=True, test_id=None, outlier_remove=True):
        super().__init__(config, test_id=test_id, load=load, outlier_remove=outlier_remove)
        self.is_train = is_train
        self.real_view_data = self.get_real_view_rays(load=load)
    
    def get_radius(self, t):
        """
        Get the radius value at time t.
        """
        return self.radius[t]
    
    def scale_intrinsics(self, intrinsics, scale):
        '''
        Scale the intrinsics matrix by a given scale factor.
        
        '''
        intrinsics = copy.deepcopy(intrinsics)
        intrinsics[..., :2, :3] *= scale
        return intrinsics

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
                           return_dirs=True,
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

        if return_dirs:
            dirs = get_view_direction(theta, phi, angle_overhead, angle_front)
        else:
            dirs = None

        return poses, dirs
    
    def update_default_view_data(self):
        print('Update Known view scale to:', self.cfg['data']['known_view_scale'])
        self.real_view_data = self.get_real_view_rays(load=True)
    
    def get_real_view_rays(self, load=True):
        if not load:
            return None
        
        H = int(self.cfg['data']['known_view_scale'] * self.H)
        W = int(self.cfg['data']['known_view_scale'] * self.W)
        
        # (3, 3)
        intri = self.scale_intrinsics(self.intrinsics, 
                                      self.cfg['data']['known_view_scale']
                                      )
        
        # (H, W, 3)
        rays_d_cam = get_camera_rays(H, W , intri[0,0], intri[1,1],
                                     intri[0,2], intri[1,2],
                                     type='OpenGL').to(torch.float32)

        t = torch.arange(self.num_frames)
        
        pose = self.poses
        
        Bs, N = pose.shape[0], H * W   
        
        
        
        
        
        rays_o = pose[...,None, None, :3, -1].repeat(1, H, W, 1)
        
        rays_d_cam = rays_d_cam[None, ...].repeat(Bs, 1, 1, 1)
        rays_d = torch.sum(rays_d_cam[..., None, :] * pose[:, None, None, :3, :3], -1)
        
        
                
        image_scale = np.stack([cv2.resize(self.images[i], (W, H), interpolation=cv2.INTER_LINEAR) for i in range(self.num_frames)], axis=0)
        depth_scale = np.stack([cv2.resize(self.depths[i], (W, H), interpolation=cv2.INTER_NEAREST) for i in range(self.num_frames)], axis=0)
        mask_scale = np.stack([cv2.resize(self.masks[i], (W, H), interpolation=cv2.INTER_NEAREST) for i in range(self.num_frames)], axis=0)
                
        
        image_scale = torch.from_numpy(image_scale).permute(0, 3, 1, 2).to(torch.float32)
        depth_scale = torch.from_numpy(depth_scale).to(torch.float32)
        mask_scale = torch.from_numpy(mask_scale).to(torch.int64)
        
        
        return {
            'rays_o': rays_o.reshape(Bs, N, 3), # torch: (num_frames, H*W, 3)
            'rays_d': rays_d.reshape(Bs, N, 3), # torch: (num_frames, H*W, 3)
            'rays_t': (t / self.num_frames)[:, None, None].repeat(1, N, 1), # torch: (num_frames, H*W)
            'rays_id': t[:, None, None].repeat(1, N, 1), # torch: (num_frames, H*W)
            'H': H, # int
            'W': W, # int
            'image': image_scale, # torch: (num_frames, 3, H, W)
            'depth': depth_scale, # torch: (num_frames, H, W)
            'mask': mask_scale, # torch: (num_frames, H, W)
            'intri': intri, # torch: (3, 3)
            'pose': pose, # torch: (num_frames, 4, 4)
            'theta': self.theta, # torch: (num_frames)
            'phi': self.phi, # torch: (num_frames)
            'radius': self.radius, # torch: (num_frames)
            
        }
    
    def sample_real_view_rays(self, idx = None, bs=1, ray_num=None):
        if idx is None:
            idx = torch.randint(0, self.num_frames, (bs,))
        elif isinstance(idx, int):
            idx = torch.tensor([idx])
        else:
            pass
        
        bs = len(idx)
        
        data = self.real_view_data
        
        samples = {}
        
        if ray_num is not None:
            index = torch.randint(0, data['H']*data['W'], (ray_num,))
            
        
        for k, v in data.items():
            if k == 'intri' or isinstance(v, float) or isinstance(v, int):
                samples[k] = v
            else:
                samples[k] = v[idx]
                
                if ray_num is not None and 'ray' in k:
                    samples[k] = samples[k][:, index]
        
        if ray_num is not None:
            samples['H'] = ray_num
            samples['W'] = 1
                    
            samples['image'] =  samples['image'].reshape(bs, 3, -1)[..., index].reshape(bs, 3, ray_num, 1)  
            samples['mask'] = samples['mask'].reshape(bs, -1)[..., index].reshape(bs, ray_num, 1)
            samples['depth'] = samples['depth'].reshape(bs, -1)[..., index].reshape(bs, ray_num, 1)                           
        
        return samples   
                  
    def get_virtual_view_data(self, size, t, 
                             keep_chirality=True,
                             camera_convention="OpenGL",
                             theta_range=[45, 105], 
                             phi_range=[0, 360], 
                             return_dirs=True, 
                             angle_overhead=30, 
                             angle_front=60,
                             uniform_sphere_rate=0.5):
    
        radius = self.get_radius(t) * self.cfg['data']['novel_view_scale_factor']
        
        theta_range = np.deg2rad(theta_range)
        phi_range = np.deg2rad(phi_range)
        angle_overhead = np.deg2rad(angle_overhead)
        angle_front = np.deg2rad(angle_front)
        
        if random.random() < uniform_sphere_rate:
            # Uniformly sample on the sphere
            
            unit_centers = F.normalize(
            torch.stack([
                torch.randn(size),
                torch.abs(torch.randn(size)),
                torch.randn(size),
            ], dim=-1), p=2, dim=1)
            
            thetas = torch.acos(unit_centers[:,1])
            
            phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
            phis[phis < 0] += 2 * np.pi
            
            cam_centers = radius * unit_centers
        
        else:
            thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
            phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
            phis[phis < 0] += 2 * np.pi

            cam_centers = torch.stack([
                radius * torch.sin(thetas) * torch.sin(phis),
                radius * torch.cos(thetas),
                radius * torch.sin(thetas) * torch.cos(phis),
            ], dim=-1) # [B, 3]

            
            
        targets = 0
                
        poses = self.get_c2w_from_cam_center(cam_centers, targets, 
                                             keep_chirality=keep_chirality, 
                                             camera_convention=camera_convention)
        
        
        if return_dirs:
            dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
        else:
            dirs = None
        
        
        # back to degree
        thetas = thetas / np.pi * 180
        phis = phis / np.pi * 180
                    
            
        
        return poses, dirs, thetas, phis, radius    
    
    def get_virtual_view_rays(self, bs=1, t=None, is_train=True, scale=None, phis=0):
        
        if t is None:
            t = torch.randint(0, self.num_frames, (bs,))
        
        elif isinstance(t, int):
            t = torch.tensor([t])
            
        
        
        if is_train:
            # poses: torch: (bs, 4, 4)
            # dirs: torch: (bs)
            # thetas: torch: (bs)
            # phis: torch: (bs)
            # radius: torch: (bs)
            poses, dirs, thetas, phis, radius = self.get_virtual_view_data(bs, t, 
                                                                    theta_range=self.cfg['data']['theta_range'],
                                                                    phi_range=self.cfg['data']['phi_range'],
                                                                    angle_overhead=self.cfg['data']['angle_overhead'],
                                                                    angle_front=self.cfg['data']['angle_front'],
                                                                    uniform_sphere_rate=self.cfg['data']['uniform_sphere_rate']
                                                                    )
        else:
            radius = self.get_radius(t)
            thetas = torch.FloatTensor([self.cfg['data']['default_polar']])
            phis = torch.FloatTensor([phis*360])
            poses, dirs = self.get_c2w_from_polar(t=t, theta=thetas, phi=phis,
                                                  angle_overhead=self.cfg['data']['angle_overhead'],
                                                  angle_front=self.cfg['data']['angle_front'])
            
        if scale is not None:
            H = int(scale* self.H)
            W = int(scale * self.W)
            intri = self.scale_intrinsics(self.intrinsics, scale=scale)
        
        else:       
            if self.is_train:
                H = int(self.cfg['data']['novel_view_scale'] * self.H)
                W = int(self.cfg['data']['novel_view_scale'] * self.W)
                intri = self.scale_intrinsics(self.intrinsics, scale=self.cfg['data']['novel_view_scale'])
            else:
                H = self.H
                W = self.W
                intri = self.intrinsics
        
        # torch: (bs, H, W, 3)
        rays_d_cam = get_camera_rays(H, W , intri[0,0], intri[1,1],
                                    intri[0,2], intri[1,2],
                                    type='OpenGL').to(torch.float32)
        
        rays_d_cam = rays_d_cam[None, ...].repeat(bs, 1, 1, 1)
        
        rays_o = poses[..., None, None, :3, -1].repeat(1, H, W, 1)
        rays_d = torch.sum(rays_d_cam[..., None, :] * poses[:, None, None, :3, :3], -1)
        
        delta_polar = thetas - self.real_view_data['theta'][t]
        delta_azimuth = phis - self.real_view_data['phi'][t]
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        
        delta_radius = radius - self.real_view_data['radius'][t]
    
                
        data = {
            'H': H,
            'W': W,
            'rays_o': rays_o.reshape(bs, -1, 3),
            'rays_d': rays_d.reshape(bs, -1, 3),
            'rays_t': (t / self.num_frames)[:, None, None].repeat(1, H*W, 1), # torch: (num_frames, H*W)
            'rays_id': t[:, None, None].repeat(1, H*W, 1), # torch: (num_frames, H*W)
            'dir': dirs,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius
        }
        return data


class RenderDataset(BaseDataset):
    def __init__(self, config, is_train=True, load=True, test_id=None, outlier_remove=False):
        super().__init__(config, test_id=test_id, load=load, outlier_remove=outlier_remove)
        self.poses_ndr, _, scale_mat = self.load_cam_params_ndr()
        if outlier_remove:
            self.poses_ndr = self.remove_outlier(self.poses_ndr, thresh=2.0)
        self.sc_ndr = scale_mat[0][0, 0]
        
        self.poses_raw, Ks, _ = self.load_cam_params_raw()
        if outlier_remove:
            self.poses = self.remove_outlier(self.poses, thresh=2.0*self.sc_ndr)
        self.K_raw = Ks[0]
        self.sc_raw = 1.0
    
    def get_align_mat(self):
        align_mat = np.eye(4)
        align_mat[1, 1] = -1.
        align_mat[2, 2] = -1.
        return align_mat
    
    
    def load_cam_params_raw(self):
        if os.path.exists(os.path.join(self.data_dir, "cameras.npz")):  # iPhone SLAM poses
            camera_dict = np.load(os.path.join(self.data_dir, "cameras.npz"))
            poses = camera_dict["c2w"]
            intrinsics = np.loadtxt(os.path.join(self.data_dir, "intrinsics.txt"))
            # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            scale_mats_np = [np.eye(4).astype(np.float32) for idx in range(self.num_frames)]

            intrinsics_all = []
            poses_all = []

            for i, pose in enumerate(poses):
                align_mat = self.get_align_mat()
                # flip the world coordinate
                # pose_align = align_mat @ pose
                poses_all.append(pose.astype(np.float32))
                intrinsics_all.append(intrinsics)
        else:
            camera_dict = np.load(os.path.join(self.data_dir, "cameras_sphere.npz"))
            world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.num_frames)]
            # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            scale_mats_np = [np.eye(4).astype(np.float32) for idx in range(self.num_frames)]

            intrinsics_all = []
            poses_all = []

            for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, _ = load_K_Rt_from_P(None, P)

                # Depth
                intrinsics_all.append(intrinsics)
                pose = np.eye(4)
                align_mat = self.get_align_mat()
                # flip the world coordinate
                pose_align = align_mat @ pose
                poses_all.append(pose_align.astype(np.float32))  # the inverse of extrinsic matrix

            intrinsics_all = np.stack(intrinsics_all)
            poses_all = np.stack(poses_all)

        return poses_all[:self.num_frames], intrinsics_all[:self.num_frames], scale_mats_np

    def load_cam_params_ndr(self):
        camera_dict = np.load(os.path.join(self.data_dir, "cameras_sphere.npz"))
        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.num_frames)]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.num_frames)]

        intrinsics_all = []
        poses_all = []

        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)

            # Depth
            intrinsics_all.append(intrinsics)
            align_mat = self.get_align_mat()
            pose_align = align_mat @ pose
            poses_all.append(pose_align.astype(np.float32))  # the inverse of extrinsic matrix

        intrinsics_all = np.stack(intrinsics_all)
        poses_all = np.stack(poses_all)

        return poses_all[:self.num_frames], intrinsics_all[:self.num_frames], scale_mats_np
    
    def load_data(self, test_id=None):
        """
        Load the dataset.

        Args:
            test_id (list): List of indices to select specific samples for testing. Default is None.
        """
        print('Real depth')
        p_images = sorted(glob(os.path.join(self.data_dir, 'rgb/*.png')))
        if len(p_images) == 0:
            p_images = sorted(glob(os.path.join(self.data_dir, 'rgb/*.jpg')))
        p_depths = sorted(glob(os.path.join(self.data_dir, 'depth/*.png')))
        p_masks = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        
        if test_id is not None:
            p_images = [p_images[i] for i in test_id]
            p_depths = [p_depths[i] for i in test_id]
            p_masks = [p_masks[i] for i in test_id]
        
        images = np.stack([cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB) for im_name in p_images]) / 255.0
        depths = np.stack([cv2.imread(im_name, cv2.IMREAD_UNCHANGED) for im_name in p_depths]) / self.cfg['data']['depth_scale']
        masks = np.stack([cv2.imread(im_name, cv2.IMREAD_UNCHANGED) for im_name in p_masks]) / 255.0
        
        return images, depths, masks

    