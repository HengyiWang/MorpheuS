import os
import copy
import glob
import open3d as o3d
import numpy as np
import cv2
import yaml
from tqdm import tqdm

import torch
import argparse
from datasets.dataset import RenderDataset
from morpheus import MorpheuS
from datasets.utils import get_camera_rays
from tools.vis import render_mesh_from_view, run_tsdf_fusion, make_video
from tools.pose_utils import creat_360_trajectory, rot_x, rot_z


class Renderer(MorpheuS):
    def __init__(self, config, backup=False, is_train=False):
        super(Renderer, self).__init__(config, backup, is_train)
        self.load_ckpt(os.path.join(self.workspace, 
                                    'models', 
                                    'model_ep_{:04d}.pth'.format(self.config['train']['n_epochs'])
                                    ))
    
    def get_dataset(self):
        '''
        return dataset
        '''
        dataset = RenderDataset(self.config)    
         
        return dataset  
    
    @torch.no_grad()
    def render_model_from_view(self, c2w, t, H=None, W=None, K=None, max_chunk=300*300, cano=False, optimize_pose=False, device=torch.device("cuda:0")):
        """
        Core function!!!
        :param c2w: [4, 4] OpenGL c2w camera pose, under reconstructed coordinate
        :param t: time stamp
        :param max_chunk:
        :param cano:
        :param optimize_pose:
        :return:
        """


        t = torch.tensor([t]).to(device)
        rays_t = (t / self.dataset.num_frames)[:, None, None].repeat(1, H*W, 1)  # [B, N, 3]
        rays_id = t[:, None, None].repeat(1, H*W, 1)

        # [H, W, 3]
        rays_d_cam = get_camera_rays(H, W,
                                     K[0, 0],
                                     K[1, 1],
                                     K[0, 2],
                                     K[1, 2],
                                     type='OpenGL').to(torch.float32).to(self.device)

        rays_d_cam = rays_d_cam.unsqueeze(0)  # [1, H, W, 3]
        rays_o = c2w[:3, -1].view(1, 1, 1, 3).repeat(1, H, W, 1)  # [1, H, W, 3]
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[None, None, None, :3, :3], -1)  # [1, H, W, 3]
        # In main script, data is obtained via sample_real_view_rays() or get_virtual_view_rays()
        rays_o = rays_o.reshape(1, -1, 3)  # [B, N, 3]
        rays_d = rays_d.reshape(1, -1, 3)  # [B, N, 3]

        shading = 'albedo'
        ambient_ratio = 1.0

        pred_rgb = []
        pred_depth = []
        N = H * W
        scale_factor = int(N // max_chunk + 1)
        if N > max_chunk:
            max_chunk = N // scale_factor + 1

        for i in range(0, N, max_chunk):
            outputs = self.render_rays(rays_o[:, i:i+max_chunk], rays_d[:, i:i+max_chunk],
                                  rays_t[:, i:i+max_chunk], rays_id[:, i:i+max_chunk],
                                  H,
                                  W,
                                  perturb=True,
                                  ambient_ratio=ambient_ratio,
                                  shading=shading,
                                  cano=cano,
                                  optimize_pose=optimize_pose)

            pred_rgb.append(outputs['image'])
            pred_depth.append(outputs['depth'])

        pred_rgb = torch.cat(pred_rgb, dim=1).reshape(1, H, W, 3)
        pred_depth = torch.cat(pred_depth, dim=1).reshape(1, H, W)

        return pred_rgb, pred_depth

    def get_recon2world_transform(self, offset=None):
        num_frames = self.dataset.num_frames
        coord_transforms = []
        for i in range(num_frames):
            c2w_raw = copy.deepcopy(self.dataset.poses_raw[i])
            c2w_ndr = copy.deepcopy(self.dataset.poses_ndr[i])

            c2w_ndr[:3, :3] /= self.dataset.sc_ndr
            coord_transform = c2w_raw @ np.linalg.inv(c2w_ndr)
            if offset is not None:
                coord_transform = coord_transform @ offset
            coord_transforms.append(coord_transform)
        return coord_transforms

    def reconstruct_bg_mesh(self, bg_mesh_path, voxel_length=0.01, depth_trunc=10.0, gray_scale=False):
        
        os.makedirs(os.path.dirname(bg_mesh_path), exist_ok=True)
        
        H, W, K = self.dataset.H, self.dataset.W, self.dataset.K_raw
        c2w_list = self.dataset.poses_raw
        rgb_list = self.dataset.images
        depth_list = self.dataset.depths
        mask_list = 1 - self.dataset.masks
        mask_list[mask_list < 0.5] = 0
        mask_list[mask_list >= 0.5] = 1
        
        run_tsdf_fusion(K, H, W, c2w_list, depth_list, rgb_list, 
                        mask_list=mask_list, save_path=bg_mesh_path, 
                        voxel_length=voxel_length, depth_trunc=depth_trunc, 
                        gray_scale=gray_scale)
        
    def reconstruct_fg_mesh(self, mesh_dir, color=True):
        self.export_all_meshes(mesh_dir, resolution=256, color=color)
        
    def render_world_video(self, mesh_dir, traj_mode, color=True, bg_smooth=True, fg_smooth=True, scale = 1.0):
        
        offset = None
        if 'frog' in self.config['exp']['exp_name']:
            offset = np.eye(4)
            offset[:3, :3] = rot_z(np.pi / 6)
            
        mesh_transforms = self.get_recon2world_transform(offset=offset)
        
        bg_mesh_path = os.path.join(self.config['data']['data_dir'], "scene_meshes", "bg_mesh.ply")
        if not os.path.exists(os.path.dirname(bg_mesh_path)):
            self.reconstruct_bg_mesh(bg_mesh_path)
        mesh_bg = o3d.io.read_triangle_mesh(bg_mesh_path)
        
        if not bg_smooth:
            mesh_bg.compute_vertex_normals()
        
        if not os.path.exists(mesh_dir):
            self.reconstruct_fg_mesh(mesh_dir, color=color) 
        mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.ply")))
            
        
        ndr2world = mesh_transforms[0]
        target = self.dataset.poses_raw[0][:3, -1] + (ndr2world[:3, :3] @ -self.dataset.poses_ndr[0][:3, -1])
        
        o2w_align = np.eye(4)
        # up_vec requires mannual tweaking
        if 'frog' in self.config['exp']['exp_name']:
            o2w_align[:3, :3] = rot_x(0. * np.pi / 180.)
        elif 'snoopy' or 'duck' or 'teddy' in self.config['exp']['exp_name']:
            o2w_align[:3, :3] = rot_x(8. * np.pi / 180.)
        elif 'mochi' in self.config['exp']['exp_name']:  # mochi
            o2w_align[:3, :3] = rot_x(15. * np.pi / 180.)
        elif 'haru' in self.config['exp']['exp_name']:
            o2w_align[:3, :3] = rot_x(45. * np.pi / 180.)
        elif 'seq028' in self.config['exp']['exp_name']:
            o2w_align[:3, :3] = rot_x(8. * np.pi / 180.)
        
        else:
            raise NotImplementedError

        
        o2w_align[:3, -1] = target.squeeze()
        o_axis = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(o2w_align)
        up_vec = o2w_align[:3, 1]  # y-axis
        
        # Create trajectory
        r = None
        if traj_mode == "real_view":
            c2w_list = self.dataset.poses_raw
        elif traj_mode == "360":
            c2w_raw0 = self.dataset.poses_raw[0]
            c2w_ref = copy.deepcopy(self.dataset.poses_raw[0])
            if 'mochi' in self.config['exp']['exp_name']:
                c2w_ref[1, -1] += 0.15
                c2w_ref[:3, :3] = c2w_ref[:3, :3] @ rot_x(-15. * np.pi / 180.)
            elif 'seq002' in self.config['exp']['exp_name']:
                c2w_ref[1, -1] += -0.12  # 0.5
                c2w_ref[:3, :3] = c2w_ref[:3, :3] @ rot_x(0.5 * np.pi / 180.)
                r = 0.7
                # c2w_ref[:3, -1] = r * c2w_ref[:3, -1] + (1 - r) * (target - c2w_ref[:3, -1])
                c2w_ref[:3, -1] = target + r * (c2w_ref[:3, -1] - target)
            elif 'seq028' in self.config['exp']['exp_name']:
                c2w_ref[1, -1] += -0.3  # 1.0
                c2w_ref[:3, :3] = c2w_ref[:3, :3] @ rot_x(5. * np.pi / 180.)
                r = 0.5
                # c2w_ref[:3, -1] = r * c2w_ref[:3, -1] + (1 - r) * (target - c2w_ref[:3, -1])
                # r = 1.75
                c2w_ref[:3, -1] = target + r * (c2w_ref[:3, -1] - target)
            
            elif 'seq004' in self.config['exp']['exp_name']:
                #c2w_ref[1, -1] += -0.3  # 1.0
                # c2w_ref[:3, :3] = c2w_ref[:3, :3] @ rot_x(5. * np.pi / 180.)
                r = 0.5
                c2w_ref[:3, -1] = r * c2w_ref[:3, -1] + (1 - r) * (target - c2w_ref[:3, -1])
                # r = 1.75
                # c2w_ref[:3, -1] = target + r * (c2w_ref[:3, -1] - target)
                
            c2w_list = creat_360_trajectory(c2w_ref, target, up_vec, self.dataset.num_frames)
        
        else:
            raise NotImplementedError()
        
        save_dir = os.path.join(self.workspace, "scene_renderings")
        os.makedirs(save_dir, exist_ok=True)
        
        save_dir_rgb = os.path.join(save_dir, "rgb")
        os.makedirs(save_dir_rgb, exist_ok=True)
        for i, mesh_file in enumerate(tqdm(mesh_files)):
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            if not fg_smooth:
                mesh.compute_vertex_normals()
            mesh.transform(mesh_transforms[i])
            
            if 'mochi' in self.config['exp']['exp_name']:
                H, W = 480, 640
                K = copy.deepcopy(self.dataset.K_raw)
                K[0, 2] = W / 2
                K[1, 2] = H / 2
            else:
                H, W = self.dataset.H, self.dataset.W
                K = copy.deepcopy(self.dataset.K_raw)

            H *= scale
            W *= scale
            if r is not None:
                K[0, 0] *= r
                K[1, 1] *= r
            K[0, :] *= scale
            K[1, :] *= scale
            c2w = c2w_list[i]
            
            img = render_mesh_from_view([mesh, mesh_bg], c2w, K, int(H), int(W), mode="color", show_backface=True)
            img = (img * 255).astype(np.uint8)
            
            save_path = os.path.join(save_dir_rgb, "{:04d}.png".format(i))
            cv2.imwrite(os.path.join(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        make_video(save_dir, save_dir_rgb, 'render_{}'.format(traj_mode))
            
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Your program description")

    parser.add_argument('--config', type=str, default='configs/snoopy.yaml', help='Path to the YAML config file')
    parser.add_argument('--traj', type=str, default='360', help='Path to the YAML config file')
    
    args = parser.parse_args()    
    config = yaml.full_load(open(args.config, 'r'))

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    renderer = Renderer(config).to(device)
    
    mesh_dir =os.path.join(renderer.workspace,  'mesh_final_color_256')
    
    renderer.render_world_video(mesh_dir, args.traj, color=True)
        

         
     
        