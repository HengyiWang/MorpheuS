import os
import sys
sys.path.append('..')
import yaml
import argparse
import numpy as np
import open3d as o3d

from preprocess import Database
from tqdm import tqdm
from tools.vis import draw_camera, gl2cv



class Visualizer:
    def __init__(self, config, align=True):
        self.data = Database(config, align=align)
        self.mesh = None
        self.vis_params = {"current_camera": None,
                           "cameras": []}

    def tsdf_fusion_poses(self, every_k=10, align_mat=None, gl_cv=True, save_name=None):
        data_dir = self.data.data_dir
        H, W, K = self.data.H, self.data.W, self.data.intrinsics.numpy()
        fx, fy, cx, cy = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]
        K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        voxel_length = 0.01
        volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=0.04,
                                                            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        
        for i in tqdm(range(self.data.n_images)):
            frame = self.data.get_frame(i)
            if i % every_k != 0:
                continue
            rgb, depth, c2w = frame["rgb"], frame["depth"], frame["c2w"]
            if align_mat is not None:
                c2w = align_mat @ c2w
            rgb = rgb * 255
            rgb = rgb.astype(np.uint8)
            rgb = o3d.geometry.Image(rgb)
            depth = depth.astype(np.float32)
            depth = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0,
                                                                    depth_trunc=8.0,
                                                                    convert_rgb_to_intensity=False)
            if gl_cv:
                c2w = gl2cv(c2w)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            w2c = np.linalg.inv(c2w)
            
            volume.integrate(rgbd, K, w2c)
        
        self.mesh = volume.extract_triangle_mesh()
        self.mesh.compute_vertex_normals()
        if save_name is not None:
            save_dir = os.path.join(data_dir, "mesh")
            os.makedirs(save_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(save_dir, '{}.ply'.format(save_name)), self.mesh)
    
    def update_pose(self, vis):
        if self.vis_params["current_camera"] is not None:
            for geom in self.vis_params["current_camera"]:
                vis.remove_geometry(geom, reset_bounding_box=False)
        c2w = self.vis_params["cameras"].pop(0)
        cam = draw_camera(c2w, cam_width=0.32, cam_height=0.24, f=0.20, color=[1, 0, 0])
        for geom in cam:
            vis.add_geometry(geom, reset_bounding_box=False)
        self.vis_params["current_camera"] = cam
        self.vis_params["cameras"].append(c2w)
        return True

    
    def visualize(self):
        self.tsdf_fusion_poses()
        
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.register_key_callback(key=ord("."), callback_func=self.update_pose)
        
        vis.add_geometry(self.mesh)
        
        # Add world coordinate axis
        world_coord_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        vis.add_geometry(world_coord_axis)
        
        # Add an unit sphere
        unit_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1., resolution=10)
        unit_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(unit_sphere)
        unit_sphere.paint_uniform_color((1, 0, 0))
        vis.add_geometry(unit_sphere)
        
        for i, c2w in enumerate(self.data.poses):
            self.vis_params["cameras"].append(c2w)
            cam = draw_camera(c2w)
            for geom in cam:
                vis.add_geometry(geom)

        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")
    parser.add_argument('--config', type=str, default='configs/snoopy.yaml', help='Path to the YAML config file')
    
    args = parser.parse_args()
    config = yaml.full_load(open(args.config, 'r'))
    
    vis = Visualizer(config)
    vis.visualize()
            