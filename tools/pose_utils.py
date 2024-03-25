import numpy as np
import copy


def safe_normalize(x, eps=1e-20):
    return x / np.sqrt(np.clip(np.sum(x * x, -1), eps, a_max=None))


def rot_x(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    rot = np.array([[1., 0., 0.],
                    [0., c, -s],
                    [0., s, c]])
    return rot


def rot_y(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    rot = np.array([[c, 0., s],
                    [0., 1., 0.],
                    [-s, 0., c]])
    return rot


def rot_z(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    rot = np.array([[c, -s, 0.],
                    [s, c, 0.],
                    [0., 0., 1.]])
    return rot


def cv2gl(c2w):
    c2w = copy.deepcopy(c2w)
    c2w[:, 1] *= -1
    c2w[:, 2] *= -1
    return c2w


gl2cv = cv2gl


def rotate_vector(rotate_axis, theta, v):
    k = safe_normalize(rotate_axis)  # unit rotate axis
    c = np.cos(theta)
    s = np.sin(theta)
    k_x_v = np.cross(k, v)
    k_dot_v = np.dot(k, v)
    v_rot = v * c + s * k_x_v + k * k_dot_v * (1 - c)
    return v_rot


def creat_360_trajectory(c2w_ref, target, rotate_axis, num_frames, reverse=False):
    """
    :param c2w_ref: reference camera pose, i.e. the starting point, under world coordinate
    :param target: target point, under world coordinate
    :param rotate_axis: rotation axis vector, under world coordinate
    :param num_frames:
    :return:
    """
    v = c2w_ref[:3, -1] - target
    x_axis, y_axis, z_axis = c2w_ref[:3, 0], c2w_ref[:3, 1], c2w_ref[:3, 2]
    if reverse:
        thetas = np.linspace(0., -2 * np.pi, num_frames)
    else:
        thetas = np.linspace(0., 2 * np.pi, num_frames)
    c2w_list = []
    for theta in thetas:
        c2w = np.eye(4)
        v_rot = rotate_vector(rotate_axis, theta, v)
        c2w[:3, -1] = v_rot + target
        # orientation is created, s.t. v under c2w_ref and v_rot under c2w are the same
        c2w[:3, 0] = rotate_vector(rotate_axis, theta, x_axis)
        c2w[:3, 1] = rotate_vector(rotate_axis, theta, y_axis)
        c2w[:3, 2] = rotate_vector(rotate_axis, theta, z_axis)
        c2w_list.append(c2w)
    return c2w_list


def create_mirror_trajectory(c2w_ref_list, target, rotate_axis):
    theta = np.pi
    c2w_list = []
    for c2w_ref in c2w_ref_list:
        v = c2w_ref[:3, -1] - target
        x_axis, y_axis, z_axis = c2w_ref[:3, 0], c2w_ref[:3, 1], c2w_ref[:3, 2]
        c2w = np.eye(4)
        v_rot = rotate_vector(rotate_axis, theta, v)
        c2w[:3, -1] = v_rot + target
        # orientation is created, s.t. v under c2w_ref and v_rot under c2w are the same
        c2w[:3, 0] = rotate_vector(rotate_axis, theta, x_axis)
        c2w[:3, 1] = rotate_vector(rotate_axis, theta, y_axis)
        c2w[:3, 2] = rotate_vector(rotate_axis, theta, z_axis)
        c2w_list.append(c2w)
    return c2w_list
