import os
import re
import copy
import imageio
import numpy as np
from PIL import Image

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
