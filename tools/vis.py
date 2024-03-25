import os
import re
import copy
import torch
import imageio
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]

def add_to_writer(writer, img, repeat=1):
    for i in range(repeat):
         writer.append_data(np.array(img))

def make_video(log_path, video_path, video_name):
    os.makedirs(log_path, exist_ok=True)
    frame_paths = sorted(os.listdir(video_path), key=alphanum_key)
    for frame in frame_paths:
        if 'png' not in frame:
            frame_paths.remove(frame)
    writer = imageio.get_writer(os.path.join(log_path, video_name+'.mp4'), fps=20)

    for i, frame_p in enumerate(frame_paths):
        img = Image.open(os.path.join(video_path, frame_p))
        add_to_writer(writer, img, repeat=1)

    writer.close()

def set_view(vis, x=0., y=0., z=0., theta_x=0., theta=0., theta_y=0):
    vis_ctr = vis.get_view_control()
    cam = vis_ctr.convert_to_pinhole_camera_parameters()
    c2w = np.array([[1., 0., 0., x],
                    [0., 1.,  0., y],
                    [0., 0,  1., z],
                    [0., 0., 0., 1.]])
    s = np.sin(theta)
    c = np.cos(theta)
    rot_z = np.array([[c, -s, 0., 0.],
                     [s, c, 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])

    s = np.sin(theta_y)
    c = np.cos(theta_y)
    rot_y = np.array([[c, 0., -s, 0.],
                     [0., 1., 0., 0.],
                     [s, 0., c, 0.],
                     [0., 0., 0., 1.]])

    s = np.sin(theta_x)
    c = np.cos(theta_x)
    rot_x = np.array([[1., 0., 0., 0.],
                     [0., c, -s, 0.],
                     [0., s, c, 0.],
                     [0., 0., 0., 1.]])
    c2w = c2w @ rot_z
    c2w = c2w @ rot_y
    c2w = c2w @ rot_x

    # This is w2c
    w2c = np.linalg.inv(c2w)
    cam.extrinsic = w2c
    vis_ctr.convert_from_pinhole_camera_parameters(cam)
    return vis_ctr

def set_c2w(vis, c2w):
    vis_ctr = vis.get_view_control()
    cam = vis_ctr.convert_to_pinhole_camera_parameters()
    # This is w2c
    w2c = np.linalg.inv(c2w)
    cam.extrinsic = w2c
    vis_ctr.convert_from_pinhole_camera_parameters(cam)

def set_K(vis, K, h=720, w=1280):
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    fx = K[0, 0]
    fy = K[1, 1]
    cx = w / 2 - 0.5
    cy = h / 2 - 0.5
    init_param.intrinsic.width = w
    init_param.intrinsic.height = h
    init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    ctr.convert_from_pinhole_camera_parameters(init_param)

def cv2gl(c2w):
    c2w = copy.deepcopy(c2w)
    c2w[:, 1] *= -1
    c2w[:, 2] *= -1
    return c2w

def gl2cv(c2w):
    return cv2gl(c2w)

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    t = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(t)

    return M_inv


def draw_zero_centred_cuboid(dims, color=[0, 1, 0]):
    x, y, z = dims
    bound = np.array([[-x/2, x/2],
                      [-y/2, y/2],
                      [-z/2, z/2]])
    return draw_cuboid(bound, color=color)


def draw_cuboid(bound, color=[0, 1, 0]):
    x_min, x_max = bound[0, 0], bound[0, 1]
    y_min, y_max = bound[1, 0], bound[1, 1]
    z_min, z_max = bound[2, 0], bound[2, 1]
    points = [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
              [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def draw_irregular_bound(points, color=[0, 1, 0]):
    # points = [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
    #           [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def draw_camera(c2w, cam_width=0.32/2, cam_height=0.24/2, f=0.10, color=[0, 1, 0], show_axis=True):
    points = [[0, 0, 0], [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
              [cam_width, cam_height, f], [-cam_width, cam_height, f]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [color for i in range(len(lines))]

    if isinstance(c2w, torch.Tensor):
        c2w = c2w.cpu().numpy()

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(c2w)

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis.scale(min(cam_width, cam_height), np.array([0., 0., 0.]))
        axis.transform(c2w)
        return [line_set, axis]
    else:
        return [line_set]


def visualize(extrinsics=None, things_to_draw=[]):

    ########################    plot params     ########################
    cam_width = 0.64/2     # Width/2 of the displayed camera.
    cam_height = 0.48/2    # Height/2 of the displayed camera.
    focal_len = 0.20     # focal length of the displayed camera.

    ########################    original code    ########################
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if extrinsics is not None:
        for c in range(extrinsics.shape[0]):
            c2w = extrinsics[c, ...]
            camera = draw_camera(cam_width, cam_height, focal_len, c2w, color=[1, 0, 0])
            for geom in camera:
                vis.add_geometry(geom)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(axis)
    for geom in  things_to_draw:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()


def set_K_c2w_arbitrary(vis, K, h, w, c2w):
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    # init_param.intrinsic.intrinsic_matrix = K
    init_param.intrinsic.width = w
    init_param.intrinsic.height = h
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    w2c = np.linalg.inv(c2w)
    init_param.extrinsic = w2c
    ctr.convert_from_pinhole_camera_parameters(init_param, allow_arbitrary=True)


def render_mesh_from_view(geom_list, c2w, K, H, W, mode="gray", show_backface=True):
    """
    :param geom_list: list of all geometries
    :param c2w: OpenCV pose, np.array [4, 4]
    :param K:
    :param H:
    :param W:
    :param mode:
    :return:
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = show_backface

    for geom in geom_list:
        vis.add_geometry(geom)

    if mode == "color":
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption(1)
    elif mode == "normal":
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption(9)
    elif mode == "gray":
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption(0)
    else:
        raise NotImplementedError

    set_K_c2w_arbitrary(vis, K, H, W, c2w)
    vis.poll_events()
    img = np.array(vis.capture_screen_float_buffer())
    vis.destroy_window()
    vis.close()

    return img


def back_proj_frame(K, H, W, c2w, depth, rgb, save_path=None, mask=None, depth_scale=1.0, depth_trunc=10.0, sdf_trunc=0.04, voxel_length=0.02, gray_scale=False, intensity_scale=1.0, alpha=0.0, save_as_pcd=False):
    """
    :param K:
    :param H:
    :param W:
    :param c2w: OpenCV c2w camera pose
    :param depth:
    :param rgb:
    :param save_path:
    :param voxel_length:
    :return:
    """

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=sdf_trunc,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    # rgb = rgb * 255
    if gray_scale:
        rgb_gray = np.mean(copy.deepcopy(rgb), axis=-1) * 255
        rgb[..., 0] = rgb_gray
        rgb[..., 1] = rgb_gray
        rgb[..., 2] = rgb_gray
    elif intensity_scale < 1.0:
        rgb *= intensity_scale
        rgb *= 255
    elif alpha > 0.0:
        rgb = rgb * alpha + (1 - alpha) * np.ones_like(rgb)
        rgb *= 255
    else:
        rgb *= 255
    rgb = rgb.astype(np.uint8)
    rgb = o3d.geometry.Image(rgb)
    depth = depth.astype(np.float32)

    if mask is not None:
        if mask.ndim == 3:
            depth[mask[:, :, 0] <= 0.] = 0.
        elif mask.ndim == 2:
            depth[mask <= 0.] = 0.
        else:
            raise NotImplementedError

    depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=depth_scale,
                                                              depth_trunc=depth_trunc,
                                                              convert_rgb_to_intensity=False)

    w2c = np.linalg.inv(c2w)
    volume.integrate(rgbd, K, w2c)
    print("Extract a triangle mesh from the volume and visualize it.")
    if not save_as_pcd:
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        if save_path is not None:
            o3d.io.write_triangle_mesh(save_path, mesh)
    else:
        mesh = volume.extract_point_cloud()
        if save_path is not None:
            o3d.io.write_point_cloud(save_path, mesh)

    return mesh


def run_tsdf_fusion(K, H, W, c2w_list, depth_list, rgb_list, mask_list=None, skip=None, save_path=None, depth_scale=1.0, depth_trunc=10.0, sdf_trunc=0.04, voxel_length=0.02, gray_scale=False):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=sdf_trunc,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    num_frames = len(c2w_list)
    for i in tqdm(range(num_frames)):
        c2w, depth, rgb = c2w_list[i], depth_list[i], rgb_list[i]

        if gray_scale:
            rgb_gray = np.mean(copy.deepcopy(rgb), axis=-1) * 255
            rgb[..., 0] = rgb_gray
            rgb[..., 1] = rgb_gray
            rgb[..., 2] = rgb_gray
        else:
            rgb *= 255
        rgb = rgb.astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)

        # mask out fore-ground
        if mask_list is not None:
            mask = mask_list[i]
            if mask.ndim == 3:
                depth[mask[:, :, 0] <= 0.] = 0.
            elif mask.ndim == 2:
                depth[mask <= 0.] = 0.
            else:
                raise NotImplementedError

        depth = depth.astype(np.float32)
        depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,
                                                                  depth,
                                                                  depth_scale=depth_scale,
                                                                  depth_trunc=depth_trunc,
                                                                  convert_rgb_to_intensity=False)

        w2c = np.linalg.inv(c2w)
        volume.integrate(rgbd, K, w2c)

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if save_path is not None:
        o3d.io.write_triangle_mesh(save_path, mesh)
    return mesh
import os
import re
import copy
import torch
import imageio
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]

def add_to_writer(writer, img, repeat=1):
    for i in range(repeat):
         writer.append_data(np.array(img))

def make_video(log_path, video_path, video_name):
    os.makedirs(log_path, exist_ok=True)
    frame_paths = sorted(os.listdir(video_path), key=alphanum_key)
    for frame in frame_paths:
        if 'png' not in frame:
            frame_paths.remove(frame)
    writer = imageio.get_writer(os.path.join(log_path, video_name+'.mp4'), fps=20)

    for i, frame_p in enumerate(frame_paths):
        img = Image.open(os.path.join(video_path, frame_p))
        add_to_writer(writer, img, repeat=1)

    writer.close()

def set_view(vis, x=0., y=0., z=0., theta_x=0., theta=0., theta_y=0):
    vis_ctr = vis.get_view_control()
    cam = vis_ctr.convert_to_pinhole_camera_parameters()
    c2w = np.array([[1., 0., 0., x],
                    [0., 1.,  0., y],
                    [0., 0,  1., z],
                    [0., 0., 0., 1.]])
    s = np.sin(theta)
    c = np.cos(theta)
    rot_z = np.array([[c, -s, 0., 0.],
                     [s, c, 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])

    s = np.sin(theta_y)
    c = np.cos(theta_y)
    rot_y = np.array([[c, 0., -s, 0.],
                     [0., 1., 0., 0.],
                     [s, 0., c, 0.],
                     [0., 0., 0., 1.]])

    s = np.sin(theta_x)
    c = np.cos(theta_x)
    rot_x = np.array([[1., 0., 0., 0.],
                     [0., c, -s, 0.],
                     [0., s, c, 0.],
                     [0., 0., 0., 1.]])
    c2w = c2w @ rot_z
    c2w = c2w @ rot_y
    c2w = c2w @ rot_x

    # This is w2c
    w2c = np.linalg.inv(c2w)
    cam.extrinsic = w2c
    vis_ctr.convert_from_pinhole_camera_parameters(cam)
    return vis_ctr

def set_c2w(vis, c2w):
    vis_ctr = vis.get_view_control()
    cam = vis_ctr.convert_to_pinhole_camera_parameters()
    # This is w2c
    w2c = np.linalg.inv(c2w)
    cam.extrinsic = w2c
    vis_ctr.convert_from_pinhole_camera_parameters(cam)

def set_K(vis, K, h=720, w=1280):
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    fx = K[0, 0]
    fy = K[1, 1]
    cx = w / 2 - 0.5
    cy = h / 2 - 0.5
    init_param.intrinsic.width = w
    init_param.intrinsic.height = h
    init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    ctr.convert_from_pinhole_camera_parameters(init_param)

def cv2gl(c2w):
    c2w = copy.deepcopy(c2w)
    c2w[:, 1] *= -1
    c2w[:, 2] *= -1
    return c2w

def gl2cv(c2w):
    return cv2gl(c2w)

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    t = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(t)

    return M_inv


def draw_zero_centred_cuboid(dims, color=[0, 1, 0]):
    x, y, z = dims
    bound = np.array([[-x/2, x/2],
                      [-y/2, y/2],
                      [-z/2, z/2]])
    return draw_cuboid(bound, color=color)


def draw_cuboid(bound, color=[0, 1, 0]):
    x_min, x_max = bound[0, 0], bound[0, 1]
    y_min, y_max = bound[1, 0], bound[1, 1]
    z_min, z_max = bound[2, 0], bound[2, 1]
    points = [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
              [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def draw_irregular_bound(points, color=[0, 1, 0]):
    # points = [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
    #           [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def draw_camera(c2w, cam_width=0.32/2, cam_height=0.24/2, f=0.10, color=[0, 1, 0], show_axis=True):
    points = [[0, 0, 0], [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
              [cam_width, cam_height, f], [-cam_width, cam_height, f]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [color for i in range(len(lines))]

    if isinstance(c2w, torch.Tensor):
        c2w = c2w.cpu().numpy()

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(c2w)

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis.scale(min(cam_width, cam_height), np.array([0., 0., 0.]))
        axis.transform(c2w)
        return [line_set, axis]
    else:
        return [line_set]


def visualize(extrinsics=None, things_to_draw=[]):

    ########################    plot params     ########################
    cam_width = 0.64/2     # Width/2 of the displayed camera.
    cam_height = 0.48/2    # Height/2 of the displayed camera.
    focal_len = 0.20     # focal length of the displayed camera.

    ########################    original code    ########################
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if extrinsics is not None:
        for c in range(extrinsics.shape[0]):
            c2w = extrinsics[c, ...]
            camera = draw_camera(cam_width, cam_height, focal_len, c2w, color=[1, 0, 0])
            for geom in camera:
                vis.add_geometry(geom)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(axis)
    for geom in  things_to_draw:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()


def set_K_c2w_arbitrary(vis, K, h, w, c2w):
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    # init_param.intrinsic.intrinsic_matrix = K
    init_param.intrinsic.width = w
    init_param.intrinsic.height = h
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    w2c = np.linalg.inv(c2w)
    init_param.extrinsic = w2c
    ctr.convert_from_pinhole_camera_parameters(init_param, allow_arbitrary=True)


def render_mesh_from_view(geom_list, c2w, K, H, W, mode="gray", show_backface=True):
    """
    :param geom_list: list of all geometries
    :param c2w: OpenCV pose, np.array [4, 4]
    :param K:
    :param H:
    :param W:
    :param mode:
    :return:
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = show_backface

    for geom in geom_list:
        vis.add_geometry(geom)

    if mode == "color":
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption(1)
    elif mode == "normal":
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption(9)
    elif mode == "gray":
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption(0)
    else:
        raise NotImplementedError

    set_K_c2w_arbitrary(vis, K, H, W, c2w)
    vis.poll_events()
    img = np.array(vis.capture_screen_float_buffer())
    vis.destroy_window()
    vis.close()

    return img


def back_proj_frame(K, H, W, c2w, depth, rgb, save_path=None, mask=None, depth_scale=1.0, depth_trunc=10.0, sdf_trunc=0.04, voxel_length=0.02, gray_scale=False, intensity_scale=1.0, alpha=0.0, save_as_pcd=False):
    """
    :param K:
    :param H:
    :param W:
    :param c2w: OpenCV c2w camera pose
    :param depth:
    :param rgb:
    :param save_path:
    :param voxel_length:
    :return:
    """

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=sdf_trunc,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    # rgb = rgb * 255
    if gray_scale:
        rgb_gray = np.mean(copy.deepcopy(rgb), axis=-1) * 255
        rgb[..., 0] = rgb_gray
        rgb[..., 1] = rgb_gray
        rgb[..., 2] = rgb_gray
    elif intensity_scale < 1.0:
        rgb *= intensity_scale
        rgb *= 255
    elif alpha > 0.0:
        rgb = rgb * alpha + (1 - alpha) * np.ones_like(rgb)
        rgb *= 255
    else:
        rgb *= 255
    rgb = rgb.astype(np.uint8)
    rgb = o3d.geometry.Image(rgb)
    depth = depth.astype(np.float32)

    if mask is not None:
        if mask.ndim == 3:
            depth[mask[:, :, 0] <= 0.] = 0.
        elif mask.ndim == 2:
            depth[mask <= 0.] = 0.
        else:
            raise NotImplementedError

    depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=depth_scale,
                                                              depth_trunc=depth_trunc,
                                                              convert_rgb_to_intensity=False)

    w2c = np.linalg.inv(c2w)
    volume.integrate(rgbd, K, w2c)
    print("Extract a triangle mesh from the volume and visualize it.")
    if not save_as_pcd:
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        if save_path is not None:
            o3d.io.write_triangle_mesh(save_path, mesh)
    else:
        mesh = volume.extract_point_cloud()
        if save_path is not None:
            o3d.io.write_point_cloud(save_path, mesh)

    return mesh


def run_tsdf_fusion(K, H, W, c2w_list, depth_list, rgb_list, mask_list=None, skip=None, save_path=None, depth_scale=1.0, depth_trunc=10.0, sdf_trunc=0.04, voxel_length=0.02, gray_scale=False):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=sdf_trunc,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    num_frames = len(c2w_list)
    for i in tqdm(range(num_frames)):
        c2w, depth, rgb = c2w_list[i], depth_list[i], rgb_list[i]

        if gray_scale:
            rgb_gray = np.mean(copy.deepcopy(rgb), axis=-1) * 255
            rgb[..., 0] = rgb_gray
            rgb[..., 1] = rgb_gray
            rgb[..., 2] = rgb_gray
        else:
            rgb *= 255
        rgb = rgb.astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)

        # mask out fore-ground
        if mask_list is not None:
            mask = mask_list[i]
            if mask.ndim == 3:
                depth[mask[:, :, 0] <= 0.] = 0.
            elif mask.ndim == 2:
                depth[mask <= 0.] = 0.
            else:
                raise NotImplementedError

        depth = depth.astype(np.float32)
        depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,
                                                                  depth,
                                                                  depth_scale=depth_scale,
                                                                  depth_trunc=depth_trunc,
                                                                  convert_rgb_to_intensity=False)

        w2c = np.linalg.inv(c2w)
        volume.integrate(rgbd, K, w2c)

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if save_path is not None:
        o3d.io.write_triangle_mesh(save_path, mesh)
    return mesh
