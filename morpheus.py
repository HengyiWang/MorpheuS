# package import
import os
import cv2
import copy
import math
import time
import yaml
import torch
import mcubes
import random
import imageio
import nerfacc
import trimesh
import argparse
import threading

import numpy as np
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from shutil import copyfile
from torch.optim import Adam
from rich.console import Console
from nerfacc import OccGridEstimator
from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef


# local import
from models.optimizer import Adan
from models.clip_encoders import ImageEncoder
from models.model import scene_representation

from datasets.dataset import DeformDataset

from tools.culling import eval_mesh, eval_depthL1
from tools.vis import make_video, set_c2w, set_K, gl2cv

from utils import sample_pdf, get_GPU_mem, custom_meshgrid
from utils import coordinates, get_sdf_loss, seed_everything, mse2psnr, safe_normalize

# Set seed in code-release version to improve reproducibility
seed_everything(2024)

class MorpheuS(nn.Module):
    def __init__(self, config, backup=True, is_train=True):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.create_log()
        self.dataset = self.get_dataset()
        self.bound, self.bounding_box = self.get_bound()
        
        self.model = self.get_model()
        
        if is_train:
            self.guidance = self.get_guidance()
            self.embeddings = self.get_embeddings(kf_every=self.config['train']['kf_every'])
        
        self.clip_encoder = self.get_clip_encoder()
        
        # Geometric initialization
        # NOTE: This is not used, but can potentially be used for initialization
        #self.geometric_init(radius=self.config['exp']['geo_radius'])
        
        self.optimizer, self.scaler, self.ema = self.get_optimizer()  
        self.occupancy_grid = self.get_occ_grid(self.aabb_train, resolution=128)
        
        if backup:
            self.file_backup()
    
    def file_backup(self):
        '''
        Backup the code and config files
        '''
        file_path_all =[ './', './models', './datasets']
        
        os.makedirs(os.path.join(self.workspace, 'recording'), exist_ok=True)
        
        for file_path in file_path_all:
            cur_dir = os.path.join(self.workspace, 'recording', file_path)
            os.makedirs(cur_dir, exist_ok=True)
            
            files = os.listdir(file_path)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(file_path, f_name), os.path.join(cur_dir, f_name))
    
    def create_log(self):
        '''
        Create log file and console
        '''
        self.epoch = 0
        self.console = Console()
        self.workspace = os.path.join(self.config['exp']['output'], self.config['exp']['exp_name'])
        os.makedirs(self.workspace, exist_ok=True)
        log_path = os.path.join(self.workspace, self.config['exp']['log'])
        self.log_ptr = open(log_path, 'a+')
        self.global_step = 0
        self.freeze_lr = True # Freeze deformation field to ensure a good initialization
    
    def get_dataset(self):
        '''
        return dataset
        '''
        dataset = DeformDataset(self.config)    
         
        return dataset     
    
    def get_bound(self):
        '''
        Create bounding box for scene representation
        '''
        bound = self.dataset.bound.to(self.device)
        bounding_box = torch.stack((self.dataset.bounding_box[:3], self.dataset.bounding_box[3:])).to(self.device)
        
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)
        
        print('Bounding_box: ', bounding_box)
        
        return bound, bounding_box

    def get_model(self):
        print('Create scene representation')
        return scene_representation(self.config, self.bound, 
                                    num_frames=self.dataset.num_frames,
                                    deform_dim=self.config['model']['deform_dim'],
                                    use_app=self.config['model']['use_app'],
                                    use_t=self.config['model']['use_t'],
                                    amb_dim=self.config['model']['amb_dim'],
                                    color_grid=self.config['model']['color_grid'],
                                    use_joint=self.config['model']['use_joint'],
                                    encode_topo=self.config['model']['encode_topo']
                                    ).to(self.device)
    
    def get_optimizer(self):
        '''
        Get optimizer, scaler and ema
        '''
        if self.config['train']['optim'] == 'adan':
            print('Use ADAN optimizer')
            # Adan usually requires a larger LR
            optimizer = Adan(self.model.get_params_all(5 * self.config['train']['lr']), 
                                  eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
            
        else: # adam
            print('Use Adam optimizer')
            optimizer = Adam(self.model.get_params_all(self.config['train']['lr']),
                                  betas=(0.9, 0.99), eps=1e-15)
            print(self.model.get_params_all(self.config['train']['lr']))
        
        scaler = torch.cuda.amp.GradScaler(enabled=self.config['exp']['fp16'])
        
        if self.config['train']['ema_decay'] > 0:
            from torch_ema import ExponentialMovingAverage
            ema = ExponentialMovingAverage(self.model.parameters(), decay=self.config['train']['ema_decay'])
        else:
            ema = None
        
        return optimizer, scaler, ema
    
    def get_clip_encoder(self):
        return ImageEncoder()
    
    def get_guidance(self):
        '''
        Load diffusion prior. Here we use Zero123. 
        NOTE: Other models can also be added here.
        NOTE: We find that using Zero123-XL may not lead to better results. You may
        need to tune guidance scale and other parameters to get better results. For
        anyone who is interested in using Zero123-XL, I would suggest to try it on 
        frog sequence, then you will see notable difference between Zero123 and Zero123-XL.
        I guess this might be because of the domain gap between the training data and 
        the real data. More synthetic data may not lead to better results in real data.
        '''
        guidance = {}
        if 'zero123' in self.config['guidance']['model']:
            print('Use zero123 guidance')
            from models.guidance.zero123_utils import Zero123
            guidance['zero123'] = Zero123(device=self.device, 
                                          fp16=self.config['exp']['fp16'],
                                          config=self.config['guidance']['zero123_config'], 
                                          ckpt=self.config['guidance']['zero123_ckpt'],
                                          vram_O=self.config['guidance']['vram_O'], 
                                          t_range=self.config['guidance']['t_range'],
                                          opt=self.config['guidance'])
        
        return guidance
    
    def get_occ_grid(self, aabb, resolution=128):
        '''
        Get occupancy grid estimator (NeRFAcc)
        '''
        return OccGridEstimator(
                roi_aabb=aabb,
                resolution=resolution)
    
    def get_kf(self, num_frames, kf_every=None):
        '''
        A very simple way to select keyframes
        NOTE: A more advanced way, for instance, select kf 
        based on relative angles, can also be used. We keep it simple here.
        '''
        frame_idx = torch.arange(0, num_frames, kf_every).long()
        
        
        if (num_frames - 1) not in frame_idx:
            frame_idx = torch.cat([frame_idx, torch.tensor([num_frames-1])])
        
        return frame_idx
    
    @torch.no_grad()
    def get_embeddings(self, kf_every=2):
        '''
        Pre-compute latent vectors for each kf
        '''
        
        # Select keyframes (a straightforward way is to select 
        # every n frames, but other strategies can also be used)
        num_frames = self.dataset.num_frames
        
        frame_idx = self.get_kf(num_frames, kf_every=kf_every)
        self.log('Use frame_idx: ', frame_idx)
        
        self.embedding_idx = frame_idx
        
        # Get angles and relative angles for weight gradient
        self.angles = torch.stack([self.dataset.get_radius(frame_idx),
                             torch.deg2rad(self.dataset.theta[frame_idx]),
                             torch.deg2rad(self.dataset.phi[frame_idx])],
                            dim=-1)
        
        self.angle_rel = self.guidance['zero123'].angle_between(self.angles, self.angles)
        
        
        
        # Get embeddings
        embeddings = {'zero123': {}}
        
        for i in frame_idx:
        
            images = self.dataset.images[i]
            masks = self.dataset.masks[i]
                    
            assert len(images.shape) == 3, 'The shape of images should be (H, W, 3)'
            assert len(masks.shape) == 2, 'The shape of images should be (H, W)'
            
            masks[masks>0.5]= 1
            masks[masks<=0.5] = 0
            
            # Masked image (Bs, C, H, W)
            masked_img = images * masks[...,None] + (1 - masks[...,None])
            masked_img = cv2.resize(masked_img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
            masked_img = torch.from_numpy(masked_img)[None, ...].permute(0,3,1,2).contiguous().to(self.device)
            
            # Compute embeddings            
            guidance_embeds = self.guidance['zero123'].get_img_embeds(masked_img)
            
            embeddings['zero123'][int(i.item())] = {
                'zero123_ws' : [1], # [1]
                'c_crossattn' : guidance_embeds[0],
                'c_concat' : guidance_embeds[1],
                'ref_polars' : [self.dataset.theta[i]], # [90]
                'ref_azimuths' : [self.dataset.phi[i]], # [0]
                'ref_radii' : [self.dataset.get_radius(i)], # [3]
            }
        
        mem = get_GPU_mem()[0]
        print(f'GPU={mem:.1f}GB.')
                
        return embeddings

    def geometric_init(self, sample_points=128, radius=0.3, chunk=1024*2048, eps=1e-8):
        '''
        Initialize the geometry to be a sphere by training the model
        NOTE: We use another initialization method, so this is not used.
        However, we keep this function here in case that it helps.
        '''

        volume = self.bounding_box[1, :] - self.bounding_box[0, :]
        center = self.bounding_box[0, :] + volume / 2
        
        print('Volume:', volume)
        print('Center:', center)

        print('geometric initialisation')
        ckpt_path = self.config['exp']['ckpt_init']
        if os.path.exists(ckpt_path):
            print('Reloading:', self.config['exp']['ckpt_init'])
            self.model.load_state_dict(torch.load(ckpt_path), strict=False)
            
        else:
            optimizer = Adam([
                                {'params': self.model.encoder.parameters(), 'weight_decay': 1e-6},
                                {'params': self.model.sigma_net.parameters(), 'eps': 1e-15}
                            ], lr=1e-3, betas=(0.9, 0.99))
            loss = 0
            pbar = tqdm(range(self.config['exp']['sphere_iters']))
            for _ in pbar:
                optimizer.zero_grad()
                coords = coordinates(sample_points - 1, device).float().t()
                pts = (coords + torch.rand_like(coords)) * volume / sample_points + self.bounding_box[0, :]


                for i in range(0, pts.shape[0], chunk):
                    optimizer.zero_grad()
                    sdf = self.model.density(pts[i:i+chunk], cano=True)['sdf'].squeeze()
                    target_sdf = (pts[i:i+chunk]- center).norm(dim=-1) - radius
                    loss = torch.nn.functional.mse_loss(sdf, target_sdf)
                    pbar.set_postfix({'loss': loss.cpu().item()})
                    
                    loss.backward()
                    optimizer.step()


                    
                if loss.item() < eps:
                    break
            self.model.zero_grad()
                
            torch.save(self.model.state_dict(), ckpt_path)
       
    def load_ckpt(self, ckpt_path):
        
        model_dict = torch.load(ckpt_path)
        
        self.model.load_state_dict(model_dict['model'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.epoch = model_dict['epoch']
        self.global_step = model_dict['global_step']
        if self.ema is not None:
            self.ema.load_state_dict(model_dict['ema'])
        
        self.scaler.load_state_dict(model_dict['scaler'])
        self.occupancy_grid.load_state_dict(model_dict['estimator'])
        
        print('Load ckpt from ', ckpt_path)
    
    def save_ckpt(self, save_path):
        
        save_dict = {}
        save_dict['model'] = self.model.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['epoch'] = self.epoch
        save_dict['global_step'] = self.global_step
        
        save_dict['ema'] = self.ema.state_dict() if self.ema is not None else None
        save_dict['scaler'] = self.scaler.state_dict()
        save_dict['estimator'] = self.occupancy_grid.state_dict()
        
        torch.save(save_dict, save_path)
        print('Save ckpt to ', save_path)
 
    def log(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush() # write immediately to file
    
    @torch.no_grad()
    def export_mesh(self, mesh_savepath, resolution=128, S=128, t=None, cano=False, color_mesh=True):
        """
        Export the mesh from our scene representation.

        Args:
            mesh_savepath (str): The file path to save the exported mesh.
            resolution (int, optional): The resolution of the mesh. Defaults to 128.
            S (int, optional): The size of each subgrid. Defaults to 128.
            t (float, optional): The timestep.
            cano (bool, optional): Whether to extract canonical shape.
            color_mesh (bool, optional): Whether to color the mesh. Defaults to True.
        """
        os.makedirs(os.path.dirname(mesh_savepath), exist_ok=True)
        
        density_thresh = 0
        sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)

        # query
        X = torch.linspace(-1, 1, resolution).split(S)
        Y = torch.linspace(-1, 1, resolution).split(S)
        Z = torch.linspace(-1, 1, resolution).split(S)

        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = self.model.density(pts.to(self.device), t=t, cano=cano)
                    sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val['sdf'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]

        print(f'[INFO] marching cubes thresh: {density_thresh} ({sigmas.min()} ~ {sigmas.max()})')

        vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1
        
        if color_mesh:
            color = self.model.density(torch.from_numpy(vertices).to(torch.float32).to(self.device), t=t, cano=cano)['albedo'].cpu().data.numpy()
        else: 
            color = None
        
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)
        mesh.export(mesh_savepath)
    
    def export_all_meshes(self, mesh_all_savepath, resolution=128, S=128, color=False): 
        """
        Export all meshes of the entire video sequence.
        """
        for i in range(self.dataset.num_frames):
            t = i / self.dataset.num_frames
            self.export_mesh(os.path.join(mesh_all_savepath, f'mesh_{self.epoch:04d}_{i:04d}.ply'), resolution=resolution, S=S, t=t, color_mesh=color)
        
    def render_all_meshes(self, mesh_dir, save_images_dir, save_video_dir, epoch, scale=4, 
                          view_360=False, video_name="video_real", save_depths_dir=None, save_video=True):
        K = copy.deepcopy(self.dataset.intrinsics)
        H, W = self.dataset.H, self.dataset.W
        H *= scale
        W *= scale
        K[0, :] *= scale
        K[1, :] *= scale
        
        video_name += "_{:04d}".format(epoch)
        depth_np = {}
        
        for i in tqdm(range(self.dataset.num_frames)):
            mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh_{:04d}_{:04d}.ply".format(epoch, i)))
            mesh = mesh.compute_vertex_normals()
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=W, height=H)
            vis.get_render_option().mesh_show_back_face = True
            vis.add_geometry(mesh)
            
            
            
            if not view_360:
                R, t = self.model.get_RT(torch.tensor([i]))
                deltaT = torch.eye(4)
                deltaT[:3, :3] = R.squeeze()
                deltaT[:3, -1] = t.squeeze()
                c2w = deltaT @ self.dataset.poses[i]
            else:
                thetas = torch.FloatTensor([self.config["data"]["default_polar"]])
                phis = torch.FloatTensor([i / self.dataset.num_frames * 360])
                c2w, _ = self.dataset.get_c2w_from_polar(t=i, theta=thetas, phi=phis)
                c2w = c2w[0]
            
            if isinstance(c2w, torch.Tensor):
                c2w = c2w.detach().cpu().numpy()
                
            set_c2w(vis, gl2cv(c2w))
            set_K(vis, K, H, W)
            vis.poll_events()
            vis.capture_screen_image(os.path.join(save_images_dir, "{:04d}.png".format(i)), False)
            
            if save_depths_dir is not None:
                vis.capture_depth_image(os.path.join(save_depths_dir, "{:04d}.png".format(i)), False)
                depth_np["depth_{}".format(i)] = np.array(vis.capture_depth_float_buffer())
            
            vis.destroy_window()
            vis.close()

        if save_video:
            make_video(save_video_dir, save_images_dir, video_name)
        if save_depths_dir is not None:
            np.savez(os.path.join(save_depths_dir, "depths.npz"), **depth_np)     
                      
    def update_learning_rate(self, scale_factor=1):
        '''
        A function that updates the learning rate
        Adapted from NDR (https://github.com/USTC3DV/NDR-code)
        '''
        print('update lr')
        if self.epoch < self.config['train']['warm_up_end']:
            if self.epoch < 100:
                learning_factor = 0.01
            else:
                learning_factor = 0.01 + (self.epoch - 100) / (self.config['train']['warm_up_end'] - 100) * 0.99
        else:
            alpha = 0.05
            progress = (self.epoch - self.config['train']['warm_up_end']) / (self.config['train']['n_epochs'] - self.config['train']['warm_up_end'])
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        learning_factor *= scale_factor

        current_learning_rate = self.config['train']['lr'] * learning_factor
        
        self.current_learning_rate = current_learning_rate
        
        for g in self.optimizer.param_groups:
            if g['name'] in ['pose']:
                g['lr'] = current_learning_rate * 1e-1
            elif g['name'] in ['encoder_sdf']:
                g['lr'] = current_learning_rate
            elif g['name'] in ['encoder_color']:
                # NOTE: We fix lr here to reduce the complexity of the code
                g['lr'] = current_learning_rate
            else:
                g['lr'] = current_learning_rate
    
    def freeze_lr_deform(self):
        '''
        Freeze the learning rate of deformation field
        (Deformation code + Deformation net + topology net)
        '''
        for g in self.optimizer.param_groups:
            if g['name'] in ['code_deform', 'decoder_deform', 'decoder_topo']:
                g['lr'] = 0.0
    
    def reset_lr_deform(self):
        for g in self.optimizer.param_groups:
            if g['name'] in ['code_deform', 'decoder_deform', 'decoder_topo']:
                g['lr'] = self.current_learning_rate
    
    def get_ortho_normal_dir(self, normals):
        '''
        Get direction that orthogonal to the normal
        '''
        n = F.normalize(normals, dim=-1)
        u = F.normalize(n[...,[1,0,2]] * torch.tensor([1., -1., 0.], device=n.device), dim=-1)
        v = torch.cross(n, u, dim=-1)
        phi = torch.rand(list(normals.shape[:-1]) + [1], device=normals.device) * 2. * np.pi
        w = torch.cos(phi) * u + torch.sin(phi) * v
        
        return w
    
    def get_normal_smoothness_loss(self, rays_o, rays_d, rays_t, depth):
        num_trunc_points = int(self.config['train']['trunc'] * 100 + 1)
                
        trunc_normal = torch.linspace(-0.5*self.config['train']['trunc'], 0.5*self.config['train']['trunc'], num_trunc_points)
        trunc_normal = trunc_normal + 0.01 * torch.rand_like(trunc_normal)
                        
        

    
        # 11, bs
        surf_pts = (depth + trunc_normal[:,None].to(depth))[..., None] * rays_d[None, ...] + rays_o[None, ...]
        surf_pts = surf_pts.view(-1, 3)

        surf_rays_t = rays_t[None, ...].repeat(num_trunc_points, 1, 1).view(-1, 1)
        surf_pts_norm = torch.linalg.norm(surf_pts, ord=2, dim=-1, keepdim=False)
        
        surf_pts = surf_pts[surf_pts_norm < 1.1]
        
        surf_normals, normal_raw = self.model.normal(surf_pts, t=surf_rays_t[surf_pts_norm < 1.1])
        
        w = self.get_ortho_normal_dir(surf_normals)
        
        surf_pts2 = surf_pts + w * self.config['train']['smoothness_std']
        surf_normals2, normal_raw = self.model.normal(surf_pts2, t=surf_rays_t[surf_pts_norm < 1.1]) 
        normal_reg = torch.mean(torch.square(surf_normals - surf_normals2))
        
        return normal_reg
    
    def render_rays(self, rays_o, rays_d, rays_t, rays_id,
                    H, W, 
                    perturb=True, 
                    bg_color=None, 
                    ambient_ratio=1.0, 
                    light_d=None,
                    shading='albedo',
                    real_view=True,
                    cano=False,
                    rays_depth=None,
                    rays_mask=None,
                    optimize_pose=False): 
        '''
        Core function for rendering
        
        Params:
            - rays_o: origin of rays [B=1, N=H*W, 3]
            - rays_d: direction of rays [B=1, N=H*W, 3]
            - rays_t: time of rays
            - rays_id: id of rays
            - H: height of the image
            - W: width of the image
            - perturb: whether to perturb points
            - bg_color: pre-computed random background color [BN, 3] in range [0, 1]
            - ambient_ratio: ambient ratio
            - light_d: light direction
            - shading: shading type
            - real_view: whether to train on observations
            - cano: whether to directly supervise canonical field (Not used)
            - rays_depth: depth of rays (None if using virtual view)
            - rays_mask: mask of rays (None if using virtual view)
            - optimize_pose: whether to optimize the camera pose
        
        Return:
            - image: rendered image [B, N, 3]
            - depth: rendered depth [B, N]
        '''
        
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        rays_t = rays_t.contiguous().view(-1, 1)
        rays_id = rays_id.contiguous().view(-1, 1)
        
        if not cano and optimize_pose:
            # Apply pose optimization
            rays_o, rays_d = self.model.pose_optimisation(rays_o, rays_d, rays_id)
        
        if rays_depth is not None:
            rays_depth = rays_depth.contiguous().view(-1, 1)
        
        if rays_mask is not None:
            rays_mask = rays_mask.contiguous().view(-1, 1)
        
        N = rays_o.shape[0]
        
        results = {}
        
        # We found this part may not improve the results
        # So we did not use it in sampling
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * t_positions
            time_step = rays_t[ray_indices]
            return self.model.density(positions, time_step, allow_shape=True, cano=cano)['sigma']
        
        
        with torch.no_grad():
            ray_indices, t_starts_, t_ends_ = self.occupancy_grid.sampling(
                rays_o,
                rays_d,
                sigma_fn=None,
                render_step_size=self.config['render']['step_size'],
                alpha_thre=0,
                stratified=True,
                cone_angle=0.0,
                early_stop_eps=0
                )
            
        if light_d is None:
            light_d = safe_normalize(rays_o + torch.randn(3, device=rays_o.device))
        
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        t_light_positions = light_d[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        xyzs = t_origins + t_dirs * t_positions
        time_step = rays_t[ray_indices]
        
        if rays_depth is not None:
            t_gt = rays_depth[ray_indices]
        
        if rays_mask is not None:
            t_mask = rays_mask[ray_indices]
        
        t_dirs = safe_normalize(t_dirs)
        
        # If no points are sampled, return a white image
        # This usually happens when target object occupies
        # a small portion of the image
        if xyzs.shape[0] == 0:
            image = torch.ones([*prefix, 3], device=rays_o.device)
            depth = torch.zeros([*prefix], device=rays_o.device)
            weights = None
            opacity = None
            normals = None
            deform = None
            normal_raw = None     
            
        else:
            sdf, sigmas, rgbs, normals, deform, normal_raw = self.model(xyzs, time_step, t_light_positions, ratio=ambient_ratio, shading=shading, cano=cano)
            
            weights, _, _ = nerfacc.render_weight_from_density(
                t_starts[..., 0],
                t_ends[..., 0],
                sigmas,
                ray_indices=ray_indices,
                n_rays=N,
            )
            
            opacity = nerfacc.accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=N)
            depth = nerfacc.accumulate_along_rays(weights, values=t_positions, ray_indices=ray_indices, n_rays=N)
            rgbs = nerfacc.accumulate_along_rays(weights, values=rgbs, ray_indices=ray_indices, n_rays=N)
                    
            if bg_color is None:
                if self.config['model']['bg_radius'] > 0 and cano and (not real_view):
                    # use the bg model to calculate bg_color
                    bg_color = self.model.background(rays_d, rays_t) # [N, 3]
                else:
                    bg_color = 1
            
            image = rgbs + (1 - opacity) * bg_color
            image = image.view(*prefix, 3)

            depth = depth.view(*prefix)
        
        results['image'] = image
        results['depth'] = depth
        results['sdf'] = sdf
        results['weights'] = weights
        results['weights_sum'] = opacity
        results['normal'] = normals
        results['deform'] = deform
        results['normal_raw'] = normal_raw
        
        if self.model.training:
            if self.config['train']['ori_weight'] > 0 and normals is not None and (not real_view):
                # orientation loss
                loss_orient = weights.detach() * (normals * t_dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.sum(-1).mean()
            
            if self.config['train']['normal_smooth_3d'] > 0 and normals is not None:
                # normal regularization 
                if self.config['train']['normal_dir']:
                    # perturb the point based on normal direction (Not used in final version)
                    w = self.get_ortho_normal_dir(normals)
                    
                    xyzs_perturb = xyzs + w * self.config['train']['smoothness_std']
                
                else:
                    # Random perturbation
                    xyzs_perturb = xyzs + torch.randn_like(xyzs) * self.config['train']['smoothness_std']                
                
                if self.config['train']['topo_none']:
                    # NOTE: topo_none is an option that here we always use the zero topo for reg
                    # We find that this way of canonical space reg can be better than the else branch
                    # for scenes without large motion
                    
                    # NOTE: Ideally, it worth trying regularization other than normal
                    # We leave this to potential future work.
                    
                    # NOTE: The reason why we did not use deformation net is because we
                    # find this might easily lead to some over-smooth, de-generated solution.
                    normals_perturb, normal_raw = self.model.normal(xyzs_perturb, topo=None, cano=cano)
                else:
                    topo = self.model.get_topo(xyzs_perturb, t=time_step)
                    normals_perturb, normal_raw = self.model.normal(xyzs_perturb, topo=topo, cano=cano)
                
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
                
                if self.config['train']['normal_smooth_3d_t'] > 0:
                    # normal smoothness with perturbation on time, easily lead to over-smooth results,
                    # so we did not use this in the final version
                    topo_t = self.model.get_topo(xyzs, t=time_step + torch.rand_like(time_step) * 1/self.dataset.num_frames)
                    normals_perturb_t, normal_raw = self.model.normal(xyzs, topo=topo_t, cano=cano)
                    results['loss_normal_perturb_t'] = (normals - normals_perturb_t).abs().mean()
                
                if self.config['train']['deform_smooth'] > 0 and not cano:
                    # deformation smoothness, would degrade results when we have large motion.
                    # Unfortunately, this is always the case for real-world data used in this paper
                    deform_perturb, topo_perturb, app_code_perturb = self.model.warp(xyzs_perturb, t=time_step)
                    results['loss_deform_perturb'] = (deform - deform_perturb).abs().mean()
            
            if (self.config['train']['deform_smooth_t'] > 0 or self.config['train']['topo_smooth_t'] > 0)  and not cano:
                # Another deformation smoothness, w/ perturbation on time, not used in final version
                deform_perturb_t, topo_perturb_t, app_code_perturb_t = self.model.warp(xyzs, t=time_step + torch.rand_like(time_step) * 1/self.dataset.num_frames)
                results['loss_deform_perturb_t'] = (deform - deform_perturb_t).abs().mean()
                results['loss_topo_perturb_t'] = (topo - topo_perturb_t).abs().mean()
            
            if self.config['train']['code_reg'] > 0 and not cano:
                # Code regularization
                # NOTE: This may not improve the quantitative results. However,
                # can avoid some tiny jittering effects.
                sample_time_step = time_step[:1]
            
                code = self.model.get_deform_code(sample_time_step)
                code_prev = self.model.get_deform_code(sample_time_step - 1/self.dataset.num_frames)
                code_next = self.model.get_deform_code(sample_time_step + 1/self.dataset.num_frames)
                results['loss_code'] = torch.square(2 * code - code_prev - code_next).mean()
            
            if (self.config['train']['normal_smooth_2d'] > 0) and (normals is not None) and (not real_view):
                
                normal_image = nerfacc.accumulate_along_rays(weights, values=(normals + 1) / 2, ray_indices=ray_indices, n_rays=N)
                results['normal_image'] = normal_image
            
            if self.config['train']['normal_smoothness'] > 0:
                # rays_o (Bs, 3)
                # rays_d (Bs, 3)
                # depth (1, Bs)
                # L_smooth, our normal smoothness loss in observation space, 
                # only apply to points that are close to the surface to save computation
                # This part is really costly, so we only sample a small number of the points
                results['normal_reg'] = self.get_normal_smoothness_loss(rays_o, rays_d, rays_t, depth)
            
            if rays_depth is not None:
                fs_loss, sdf_loss = get_sdf_loss(t_positions, t_gt, sdf, self.config['train']['trunc'], mask = t_mask)
                results['sdf_loss'] = sdf_loss
                results['fs_loss'] = fs_loss
            else:
                fs_loss, sdf_loss = torch.tensor(0.0, device=rays_o.device), torch.tensor(0.0, device=rays_o.device)
        
        return results

    def progressive_view(self, exp_iter_ratio):
        '''
        Progressive view expansion. We found this might not improve the results.
        '''
        
        if self.config['train']['progressive_view']:
            r = min(1.0, self.config['train']['progressive_view_init_ratio'] + 2.0*exp_iter_ratio)
            self.config['data']['phi_range'] = [self.config['data']['default_azimuth'] * (1 - r) + self.config['data']['full_phi_range'][0] * r,
                                  self.config['data']['default_azimuth'] * (1 - r) + self.config['data']['full_phi_range'][1] * r]
            self.config['data']['theta_range'] = [self.config['data']['default_polar'] * (1 - r) + self.config['data']['full_theta_range'][0] * r,
                                    self.config['data']['default_polar'] * (1 - r) + self.config['data']['full_theta_range'][1] * r]
    
    def progressive_level(self, exp_iter_ratio):
        '''
        Coarse-to-fine training
        '''
        if self.config['train']['progressive_level']:
                self.model.max_level = min(1.0, 0.5 + 0.5*exp_iter_ratio)
        
    def sample_view(self, real_view, cano):
        '''
        Sample view for training
        
        return:
            - data: sampled view (Virtual view or real view)
        '''
        if real_view and cano: # Not used
            # Use first frame as canonical shape
            data = self.dataset.sample_real_view_rays(idx=0)
        elif real_view and (not cano):
            data = self.dataset.sample_real_view_rays(ray_num=2048)
        elif (not real_view) and cano: # Not used
            data = self.dataset.get_virtual_view_rays(t=0)
        else:
            data = self.dataset.get_virtual_view_rays()
        
        return data
        
    def get_rays_from_data(self, data, real_view):
        
        '''
        Get rays from view data.Return extra data 
        (depth + mask) for real view
        '''
        
        rays_o = data['rays_o'].to(self.device) # [B, N, 3]
        rays_d = data['rays_d'].to(self.device) # [B, N, 3]
        rays_t = data['rays_t'].to(self.device) # [B, N, 1]
        rays_id = data['rays_id'].to(self.device) # [B, N, 1]
        
        B, N = rays_o.shape[:2]
        
        if real_view:
            rays_depth = data['depth'].view(B, -1, 1).to(self.device) # [B, N, 1]
            rays_mask = data['mask'].view(B, -1, 1).to(self.device) # [B, N, 1]
            
        else:
            rays_depth = None
            rays_mask = None
        
        # Add small noise to the real view rays, outdated option
        # Keep it here for potential future use
        if self.config['train']['real_view_noise'] > 0:
            rays_o = rays_o + torch.randn(3, device=self.device) * self.config['train']['real_view_noise']
            rays_d = rays_d + torch.randn(3, device=self.device) * self.config['train']['real_view_noise']
        
        return rays_o, rays_d, rays_t, rays_id, rays_depth, rays_mask

    def get_shading(self, exp_iter_ratio, real_view):
        '''
        Get shading model for rendering
        '''
        
        if real_view:
            ambient_ratio = 1.0 # Real view would use albedo only 
            shading = 'albedo_normal'  # return normal as well, but still albedo as ambient_ratio=1.0
        else:
            if exp_iter_ratio <= self.config['train']['albedo_iter_ratio']: # default 0.0
                ambient_ratio = 1.0
                shading = 'albedo'
                
            # Random shading    
            else:
                min_ambient_ratio = self.config['train']['min_ambient_ratio']
                ambient_ratio =  min_ambient_ratio + (1.0-min_ambient_ratio) * random.random()
                
                if random.random() >= (1.0 - self.config['train']['textureless_ratio']):
                    shading = 'textureless' # color = lambertian, nothing to do with albedo
                else:
                    shading = 'lambertian'
        
        return ambient_ratio, shading
    
    def get_bg_color(self, real_view, B=None, N=None):
        '''
        Get background color for rendering
        '''
        if real_view:
            bg_color = torch.rand((B * N, 3), device=self.device)
        
        else:
            # random background
            if self.config['model']['bg_radius'] > 0 and random.random() > 0.5:
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg
        
        return bg_color
    
    def update_occ_grid(self, rays_t, cano):
        '''
        Update occ grid for rendering
        '''
    
        def occ_eval_fn(x):
            return self.model.density(x, rays_t, allow_shape=True, cano=cano)['sigma'] * self.config['render']['step_size'] # step size        
        
        self.occupancy_grid.update_every_n_steps(step=self.global_step-1, occ_eval_fn=occ_eval_fn)
    
    def get_pred_from_outputs(self, outputs, B, H, W):
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
        else:
            pred_normal = None
        
        pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
        pred_sdf = outputs['sdf']
        
        return pred_rgb, pred_depth, pred_mask, pred_normal, pred_sdf
        
    def get_gt_from_data(self, data, bg_color, B, H, W):
        gt_rgb = data['image'].to(self.device)   # [B, 3, H, W]
        gt_depth = data['depth'].to(self.device)   # [B, H, W]
        gt_mask = data['mask'].to(self.device) # [B, H, W]
        
        if len(gt_rgb.shape) == 3:
            gt_mask = gt_mask.unsqueeze(0)
            gt_rgb = gt_rgb.unsqueeze(0)
            gt_depth = gt_depth.unsqueeze(0)
        
        gt_mask[gt_mask>0.5] = 1.0
        gt_mask[gt_mask<=0.5] = 0.0
        
        gt_rgb = gt_rgb * gt_mask[:, None].float() + bg_color.reshape(B, H, W, 3).permute(0,3,1,2).contiguous() * (1 - gt_mask[:, None].float())

        return gt_rgb, gt_depth, gt_mask
        
    def get_real_view_render_loss(self, pred_rgb, pred_depth, pred_mask,
                                  gt_rgb, gt_depth, gt_mask,
                                  rays_o, rays_d):
        
        loss = 0
        
        # Color loss
        if self.config['train']['rgb_weight'] > 0:
            color_loss = self.config['train']['rgb_weight'] * F.mse_loss(pred_rgb, gt_rgb)
            loss += color_loss
        
        # Mask loss
        if self.config['train']['mask_weight'] > 0:
            mask_loss = self.config['train']['mask_weight'] * F.binary_cross_entropy(pred_mask[:, 0].clip(1e-5, 1.0 - 1e-5), gt_mask.float()) #F.mse_loss(pred_mask[:, 0], gt_mask.float())
            loss += mask_loss
        
        # Depth loss
        if self.config['train']['depth_weight'] > 0:
            # Handle outliers
            depth_mask = torch.ones_like(gt_depth)
            depth_mask[gt_depth<=0] = 0
            

            xyzs = rays_o + gt_depth.reshape(1, -1, 1) * rays_d
                            
            pts_norm = torch.linalg.norm(xyzs, ord=2, dim=-1, keepdim=True)
            
            outside = pts_norm > 1.1
            depth_mask[outside.view(*depth_mask.shape)] = 0
            depth_mask[gt_mask<=0.5] = 0
            
            valid_gt_depth = gt_depth * depth_mask # [B,]    
            valid_pred_depth = pred_depth[:, 0] * depth_mask# [B,]
            
            depth_loss = self.config['train']['depth_weight'] * F.mse_loss(valid_pred_depth, valid_gt_depth)
            loss += depth_loss
        
        return loss
        
    def get_real_view_point_loss(self, gt_rgb, gt_depth, gt_mask, 
                                 rays_o, rays_d, rays_t, outputs):
        
        loss = 0
        # SDF loss
        if self.config['train']['sdf_weight'] > 0:
            loss += self.config['train']['sdf_weight'] * outputs['sdf_loss']
        
        # SDF regularization
        if self.config['train']['sdf_reg'] > 0:
            loss += self.config['train']['sdf_reg'] * torch.mean(pred_sdf**2)
        
        # Free space loss
        if self.config['train']['fs_weight'] > 0:
            loss += self.config['train']['fs_weight'] * outputs['fs_loss']
        
        if self.config['train']['surf_sdf_weight'] > 0:
                
            depth_mask = torch.ones_like(gt_depth)
            
            depth_mask[gt_depth<=0] = 0
            

            xyzs = rays_o + gt_depth.reshape(1, -1, 1) * rays_d
                            
            pts_norm = torch.linalg.norm(xyzs, ord=2, dim=-1, keepdim=True)
            # # Denoise. not use: out of mask or sphere
            outside = pts_norm > 1.1
            depth_mask[outside.view(*depth_mask.shape)] = 0
            depth_mask[gt_mask<=0.5] = 0
            
            results = self.model.density(xyzs.reshape(-1, 3), t=rays_t.reshape(-1, 1))
            sdf = results['sdf']
            albedo = results['albedo']
            
            masked_sdf = sdf.view(*depth_mask.shape)[depth_mask.to(torch.bool)]                
            masked_color = albedo.view(*depth_mask.shape, 3).permute(0, 3, 1, 2).contiguous()
            
            
            surf_color_loss = self.config['train']['surf_color_weight'] * F.mse_loss(masked_color * depth_mask[None, ...], gt_rgb * depth_mask[None, ...])
            
            surf_sdf_loss = self.config['train']['surf_sdf_weight'] * F.mse_loss(masked_sdf, torch.zeros_like(masked_sdf), reduction='mean')
            loss +=  surf_sdf_loss + surf_color_loss
        
        return loss
          
    def get_nearest_frame(self, rays_id):
        frame_id = rays_id[0, 0, 0].cpu()
                                
        abs_diff = torch.abs(self.embedding_idx - frame_id)

        # Find the index of the minimum absolute difference.
        nearest_index = torch.argmin(abs_diff)

        # Use the nearest_index to index your dictionary.
        nearest_frame = self.embedding_idx[nearest_index].item()
        
        return nearest_index, nearest_frame
        
    def get_virtual_view_loss(self, pred_rgb, polar, azimuth, radius, nearest_frame, guidance_path=None):
        
        # Use current frame or first frame
        # Since first frame usually has a good mask
        if self.config['guidance']['zero123_train'] == 'cur_or_one':
            if random.random() > 0.5:
                loss_cur, t_cur, scale_cur, noise_cur = self.guidance['zero123'].train_step(self.embeddings['zero123'][nearest_frame], 
                                                            pred_rgb, polar, azimuth, radius, 
                                                            guidance_scale=self.config['guidance']['zero123_guidance_scale'],
                                                            as_latent=False, 
                                                            grad_scale=self.config['guidance']['zero123_grad_weight'], 
                                                            save_guidance_path=guidance_path)
                loss_guidance = loss_cur
            
            else:
                polar_t = polar + self.embeddings['zero123'][nearest_frame]['ref_polars'][0]
                azimuth_t = azimuth + self.embeddings['zero123'][nearest_frame]['ref_azimuths'][0]
                radius_t = radius + self.embeddings['zero123'][nearest_frame]['ref_radii'][0]
                
                target_id = 0
                target_frame_id = self.embedding_idx[target_id].item()
                        
                polar_k = polar_t - self.embeddings['zero123'][target_frame_id]['ref_polars'][0]
                azimuth_k = azimuth_t - self.embeddings['zero123'][target_frame_id]['ref_azimuths'][0]
                radius_k = radius_t - self.embeddings['zero123'][target_frame_id]['ref_radii'][0]
                                    
                azimuth_k[azimuth_k > 180] -= 360
                
                loss_1, t_1, scale_1, noise_1 = self.guidance['zero123'].train_step(self.embeddings['zero123'][target_frame_id], 
                                                                        pred_rgb, polar_k, azimuth_k, radius_k, 
                                                                        guidance_scale=self.config['guidance']['zero123_guidance_scale'],
                                                                        as_latent=False, 
                                                                        grad_scale=self.config['guidance']['zero123_grad_weight'], 
                                                                        save_guidance_path=guidance_path)

                loss_guidance = loss_1
        else:
            # NOTE: We have tried various ways for this, but none of them seem to be able
            # to significantly improve the results without improving the computational complexity. 
            # (e.g, interpolation, keyframes that have largest relative angle with respect to 
            # current frame, etc)
            raise NotImplementedError()
            
        
        return loss_guidance

    def get_regularization_loss(self, outputs, pred_normal, cano=False):
        
        loss = 0
        
        if self.config['train']['entropy_weight'] > 0:
            alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
            loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            lambda_entropy = self.config['train']['entropy_weight'] * min(1, 2 * self.global_step / self.config['exp']['end_iter'])
            loss = loss + lambda_entropy * loss_entropy
        
        if self.config['train']['normal_smooth_2d'] > 0 and pred_normal is not None:
            loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + \
                            (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
            loss = loss + self.config['train']['normal_smooth_2d']  * loss_smooth
        
        if self.config['train']['ori_weight'] > 0 and 'loss_orient' in outputs:
            loss_orient = outputs['loss_orient']
            loss = loss + self.config['train']['ori_weight'] * loss_orient
            
        if self.config['train']['normal_smooth_3d'] > 0 and 'loss_normal_perturb' in outputs:
            loss_normal_perturb = outputs['loss_normal_perturb']
            loss = loss + self.config['train']['normal_smooth_3d'] * loss_normal_perturb
        
        if self.config['train']['normal_smooth_3d_t'] > 0 and 'loss_normal_perturb_t' in outputs:
            loss_normal_perturb_t = outputs['loss_normal_perturb_t']
            
            loss = loss + self.config['train']['normal_smooth_3d_t'] * loss_normal_perturb_t
                            
        # eikonal
        if outputs['normal_raw'] is not None and self.config['train']['eik_weight'] > 0:
            normal = outputs['normal_raw']
            gradient_error = (torch.linalg.norm(normal, ord=2, dim=-1) - 1.0) ** 2
            loss = loss + self.config['train']['eik_weight'] * torch.mean(gradient_error)
        
        if self.config['train']['beta_weight'] > 0:
            loss = loss + self.config['train']['beta_weight'] * torch.mean(self.model.sdf2density.get_beta())
        
        if self.config['train']['normal_smoothness'] > 0:
            loss = loss + self.config['train']['normal_smoothness'] * outputs['normal_reg']
            
        if self.config['train']['deform_weight'] > 0:
            loss = loss + self.config['train']['deform_weight'] * outputs['deform'].abs().mean()
        
        if self.config['train']['deform_smooth'] > 0 and 'loss_deform_perturb' in outputs:
            loss = loss + self.config['train']['deform_smooth'] * outputs['loss_deform_perturb']
        
        if self.config['train']['deform_smooth_t'] > 0 and 'loss_deform_perturb_t' in outputs:
            loss = loss + self.config['train']['deform_smooth_t'] * outputs['loss_deform_perturb_t']
            
        if self.config['train']['topo_smooth_t'] > 0 and 'loss_topo_perturb_t' in outputs:
            loss = loss + self.config['train']['topo_smooth_t'] * outputs['loss_topo_perturb_t']
            
        if self.config['train']['code_reg'] > 0 and not cano and 'loss_code' in outputs:
            loss = loss + self.config['train']['code_reg'] * outputs['loss_code']
        
        return loss     
        
    def train_step(self, real_view=True, cano=False, optimize_pose=False):
        '''
        Core function for training the model
        
        Params:
            - real_view: whether to sample from real observation.
            - cano: whether to directly supervise canonical field.
              NOTE: This is an outdated option during the development of 
              MorpheuS and we do not use it in the final version. The reason
              to not use it is that we find that canonical field is more like
              a continous field instead of a simple shape, directly supervise
              it may not lead to a good canonical field for the entire sequence.
              This might worth explore it further so we keep this option here.
            - optimize_pose: whether to optimize the camera pose.
        
        '''
        
        # Coarse-to-fine training, set up view range +
        # Freq bands & feature grid resolutions
        
        exp_iter_ratio = self.epoch / self.config['train']['n_epochs']
        self.progressive_view(exp_iter_ratio)
        self.progressive_level(exp_iter_ratio)
        
        # Sample rays for training
        # No virtual view during initial warm-up
        # TODO: Check if we can kick this out as we have lr warm-up already
        if self.global_step < self.config['train']['warm_up_steps']:
            real_view=True
        
        # Sample data (Real view or virtual view or cano view)
        data = self.sample_view(real_view, cano)
        rays_o, rays_d, rays_t, rays_id, rays_depth, rays_mask = self.get_rays_from_data(data, real_view)
        
        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        
        # Get shading model & background color
        ambient_ratio, shading = self.get_shading(exp_iter_ratio, real_view)
        bg_color = self.get_bg_color(real_view, B=B, N=N)
        
        # Update the occupancy grid in NeRFAcc
        self.update_occ_grid(rays_t, cano)
        
        outputs = self.render_rays(rays_o, rays_d, rays_t, rays_id, H, W, 
                                   perturb=True, bg_color=bg_color,
                                   ambient_ratio=ambient_ratio,
                                   shading=shading, real_view=real_view,
                                   cano=cano, rays_depth=rays_depth, rays_mask=rays_mask, 
                                   optimize_pose=optimize_pose)

        pred_rgb, pred_depth, pred_mask, pred_normal, pred_sdf = self.get_pred_from_outputs(outputs, B, H, W)
         
        
        loss = 0
        
        # Real view loss
        if real_view:
            gt_rgb, gt_depth, gt_mask = self.get_gt_from_data(data, bg_color, B, H, W)
            
            loss += self.get_real_view_render_loss(pred_rgb, pred_depth, pred_mask,
                                                   gt_rgb, gt_depth, gt_mask,
                                                   rays_o, rays_d)
            
            loss += self.get_real_view_point_loss(gt_rgb, gt_depth, gt_mask,
                                                  rays_o, rays_d, rays_t, outputs)
            
        # Virtual view loss
        else:            
            
            polar = data['polar'].to(self.device)
            azimuth = data['azimuth'].to(self.device)
            radius = data['radius'].to(self.device)
            
            if self.config['exp']['save_guidance'] and (self.global_step % self.config['exp']['save_guide_intervel'] == 0):
                guidance_path = os.path.join(self.workspace, 'guidance', '{:06d}_zero123.png'.format(self.global_step))
                os.makedirs(os.path.dirname(guidance_path), exist_ok=True)
            else:
                guidance_path = None
            
            
            nearest_index, nearest_frame = self.get_nearest_frame(rays_id)
                
                
            loss += self.get_virtual_view_loss(pred_rgb, polar, azimuth, radius, nearest_frame, guidance_path=guidance_path)
        
        # Regularization loss
        loss += self.get_regularization_loss(outputs, pred_normal, cano=cano)
        
        return pred_rgb, pred_depth, loss
    
    def eval_step(self, data, cano=False, optimize_pose=False, max_chunk=300*300):
        rays_o, rays_d, rays_t, rays_id, rays_depth, rays_mask = self.get_rays_from_data(data, False)
        
        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        
        pred_rgb = []
        pred_depth = []
        
        scale_factor = int(N//max_chunk+1)
        
        if N > max_chunk:
            max_chunk = N // scale_factor + 1
        
        
        for i in range(0, N, max_chunk):
            outputs = self.render_rays(rays_o[:, i:i+max_chunk], rays_d[:, i:i+max_chunk], 
                                       rays_t[:, i:i+max_chunk], rays_id[:, i:i+max_chunk], 
                                       H, W, perturb=True, ambient_ratio=ambient_ratio, 
                                       shading=shading, cano=cano, optimize_pose=optimize_pose)
            
            pred_rgb.append(outputs['image'])
            pred_depth.append(outputs['depth'])
        
        
        pred_rgb = torch.cat(pred_rgb, dim=1).reshape(B, H, W, 3)
        pred_depth = torch.cat(pred_depth, dim=1).reshape(B, H, W)
        
        return pred_rgb, pred_depth
        
    def get_render_data(self, i, cano=False, real_view=False, view_360=False, phis=0, scale=1.0):
        if cano:
            # 360 degree view of the canonical shape
            data = self.dataset.get_virtual_view_rays(t=0, is_train=False, phis=i/self.dataset.num_frames)
        elif real_view:
            data = self.dataset.sample_real_view_rays(idx=i)
        elif view_360:
            data = self.dataset.get_virtual_view_rays(t=i, is_train=False, phis=i/self.dataset.num_frames, scale=scale)
        else:
            data = self.dataset.get_virtual_view_rays(t=i, is_train=False, phis=phis)
            
        return data
      
    @torch.no_grad()      
    def render_test_video(self, write_video=True, phis=0, fps=25, scale=1.0,
                          cano=False, real_view=False, view_360=False, eval_clip=False,
                          test_name='test'): 
        
        
        name = f'{test_name}_ep{self.epoch:04d}'  
        save_path = os.path.join(self.workspace, 'results')
        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")
        
        pbar = tqdm(total=self.dataset.num_frames, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if write_video:
            all_preds = []
            all_preds_depth = []
            
        clip_total = 0
            
        
        for i in range(self.dataset.num_frames):
            data = self.get_render_data(i, cano=cano, real_view=real_view, phis=phis, view_360=view_360, scale=scale)
            
            with torch.cuda.amp.autocast(enabled=self.config['exp']['fp16']):
                preds, preds_depth = self.eval_step(data, cano=cano, 
                                                    optimize_pose=True if real_view else False)
            
            pred = preds[0].detach().cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            
            pred_depth = preds_depth[0].detach().cpu().numpy()
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
            pred_depth = (pred_depth * 255).astype(np.uint8)
            
            
            Bs, H, W, c = preds.shape
            
            if write_video:
                if H % 2 == 1 or W % 2 == 1:
                    pred = cv2.resize(pred, (W*2, H*2))
                    pred_depth = cv2.resize(pred_depth, (W*2, H*2))
                    
                all_preds.append(pred)
                all_preds_depth.append(pred_depth)
            else:
                cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
            
            
            # CLIP evaluation
            if eval_clip:
                gt_data = self.get_render_data(i, real_view=True)
                
                gt = gt_data['image']
                gt_mask = gt_data['mask'].float()
                
                gt_mask[gt_mask>0.5] = 1.0
                gt_mask[gt_mask<=0.5] = 0.0
                
                image_pred = preds.permute(0, 3, 1, 2)
                image_gt = (gt*gt_mask.float()).to(preds.device) + 1- gt_mask.float().to(preds.device)  
                if H != 224 or W != 224:
                    image_pred = F.interpolate(image_pred, (224, 224), mode="bilinear")
                    image_gt = F.interpolate(image_gt, (224, 224), mode="bilinear")
                score = self.clip_encoder.get_similarity_from_image(image_pred, image_gt)
                
                clip_total += score.item()
                
            
            pbar.update(1)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=fps, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=fps, quality=8, macro_block_size=1)
        
        if self.ema is not None:
            self.ema.restore()
        
        clip_avg = clip_total / self.dataset.num_frames
        
        if eval_clip:
            self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}]  ClIP={clip_avg}...")
        self.log(f"==> Finished Test.")
                
    def train_one_epoch(self, max_epochs, n_iters=10):
        '''
        Train MorpheuS for one epoch
        '''
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.config['exp']['exp_name']} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} , beta={self.model.sdf2density.get_beta().data.cpu().numpy():.3f},......")
        
        total_loss = 0
        pbar = tqdm(total=n_iters, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        self.model.train()
        self.occupancy_grid.train()
        self.local_step = 0
        self.optimizer.zero_grad()
                
        for _ in range(n_iters):
            with torch.cuda.amp.autocast(enabled=self.config['exp']['fp16']):
                for _ in range(self.config['train']['virtual_freq']):
                    if self.freeze_lr:
                        self.freeze_lr_deform()
                    self.global_step += 1
                    self.local_step  += 1
                    
                    pred_rgbs, pred_depths, loss = self.train_step(real_view=False, cano=False, optimize_pose=False)
                    self.scaler.scale(1/self.config['train']['virtual_freq'] * loss).backward()
                    
                    
                    if self.freeze_lr:  
                        # Freeze deform learning rate, need update then run real_view                    
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        self.optimizer.zero_grad()
                        self.model.zero_grad()
                
                if self.freeze_lr:
                    # Reset learning rate for real view
                    self.reset_lr_deform()
                
                for _ in range(self.config['train']['real_freq']):
                    self.global_step += 1
                    self.local_step  += 1
                    pred_rgbs, pred_depths, loss = self.train_step(real_view=True, cano=False, optimize_pose=True)
                    self.scaler.scale(loss).backward()                        
                
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
            
            loss_val = loss.item()
            total_loss += loss_val
                        
            pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
            pbar.update(1)
        
        if self.ema is not None:
            self.ema.update()
                    
    def train(self, max_epochs):
        '''
        Train MorpheuS
        '''
        
        # Save the mesh after geometric initialization
        # Check it to see if you get a good initial shape :)
        # A bad initial shape may lead to a bad final shape :(
        self.export_mesh(os.path.join(self.workspace, 'mesh', f'init.ply'))
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.update_learning_rate()
            
            self.train_one_epoch(max_epochs)
            
            # Change configuration for next epoch
            if self.epoch > self.config['train']['freeze_epoch']:
                self.freeze_lr = False
            
            if self.epoch > 200 + self.config['train']['warm_up_end']:
                start = 200 + self.config['train']['warm_up_end']
                
                
                end_t = 0.02 + 0.48 * (1- (self.epoch -  start) / (max_epochs - start))
                
                self.guidance['zero123'].update_t_range([0.02, end_t])
                self.config['train']['ori_weight'] = 0.002
                self.config['train']['rgb_weight'] = 10.0
                
                #TODO: Remove this as well?
                self.config['train']['beta_weight'] = 0.3 
            
            if self.epoch > 800:
                self.config['data']['novel_view_scale'] = self.config['data']['novel_view_scale_final']
            
            if self.epoch == max_epochs:
                path_to_save =  os.path.join(self.workspace, 'models')
                os.makedirs(path_to_save, exist_ok=True)
                self.save_ckpt(os.path.join(path_to_save, 'model_ep_{:04d}.pth'.format(self.epoch)))
            
            # Render test video
            if self.epoch % self.config['exp']['test_interval'] == 0 or self.epoch==max_epochs:
                self.render_test_video(phis=0, test_name='test')
                self.render_test_video(phis=0.5, test_name='test_180')
                self.render_test_video(cano=True, test_name='test_cano')
                self.render_test_video(view_360=True, test_name='test_360', eval_clip=True)
                self.render_test_video(real_view=True, test_name='test_real')
            
            if self.epoch % self.config['exp']['mesh_interval'] == 0 or self.epoch==max_epochs:
                self.export_mesh(os.path.join(self.workspace, 'mesh', f'mesh_{self.epoch:04d}.ply'), cano=True)
            
            if self.epoch % self.config['exp']['mesh_all_interval'] == 0 or self.epoch==max_epochs:
                
                mesh_all_dir = os.path.join(self.workspace, 'mesh_all')
                resolution = 128 if self.epoch != max_epochs else 256
                self.export_all_meshes(mesh_all_dir, resolution=resolution)
                
                images_real_dir = os.path.join(self.workspace, 'images_real', f'image_{self.epoch:04d}')
                images_360_dir = os.path.join(self.workspace, 'images_360', f'image_{self.epoch:04d}')
                video_dir = os.path.join(self.workspace, 'videos')
                depth_dir = os.path.join(self.workspace, "depths", f'depths_{self.epoch:04d}')
                target = f'mesh_{self.epoch:04d}'
                
                os.makedirs(images_real_dir, exist_ok=True)
                os.makedirs(images_360_dir, exist_ok=True)
                os.makedirs(video_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)
            
                self.render_all_meshes(mesh_all_dir, images_real_dir, video_dir, self.epoch, 
                                       scale=1, save_depths_dir=depth_dir, save_video=False)
                
                self.render_all_meshes(mesh_all_dir, images_real_dir, video_dir, self.epoch)
                
                self.render_all_meshes(mesh_all_dir, images_360_dir, video_dir, self.epoch, 
                                       view_360=True, video_name='video_360')
                
                if self.epoch % self.config['exp']['mesh_all_eval_interval'] == 0 or self.epoch==max_epochs:  
                    t1 = threading.Thread(target=eval_mesh, args=(self.workspace, mesh_all_dir, self.dataset, target, epoch))
                    t2 = threading.Thread(target=eval_depthL1, args=(depth_dir, self.dataset))
                    t1.start()
                    t2.start()
            
        t1.join()
        t2.join()
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Your program description")

    parser.add_argument('--config', type=str, default='configs/kfusion/frog.yaml', help='Path to the YAML config file')

    args, remaining_args = parser.parse_known_args()
    
    config = yaml.full_load(open(args.config, 'r'))

    subparsers = parser.add_subparsers(dest='section', help='Config section')

    for section_name, section_params in config.items():
        subparser = subparsers.add_parser(section_name)
        for key, value in section_params.items():
            subparser.add_argument(f'--{key}', default=value, type=type(value))

    args = parser.parse_args(remaining_args)

    if args.section in config:
        for key, value in vars(args).items():
            if key != 'section' and value is not None:
                config[args.section][key] = value
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    morpheus = MorpheuS(config).to(device)

    with open(os.path.join(morpheus.workspace, 'config.yaml'), "w") as config_file:
        yaml.dump(config, config_file)
    
    morpheus.train(config['train']['n_epochs'])
        

    





















