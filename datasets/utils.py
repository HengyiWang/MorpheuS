import cv2
import torch
import numpy as np

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def get_camera_rays(H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    
    i = i.to(fx.device)
    j = j.to(fx.device)
    
    
    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------

    if cx is None:
        cx, cy = 0.5 * W, 0.5 * H

    if fy is None:
        fy = fx
    if type == 'OpenGL':
        dirs = torch.stack([(i + 0.5 - cx)/fx, -(j + 0.5 - cy)/fy, -torch.ones_like(i, device=fx.device)], -1)
    elif type == 'OpenCV':
        dirs = torch.stack([(i + 0.5 - cx)/fx, (j + 0.5 - cy)/fy, torch.ones_like(i,  device=fx.device)], -1)
    else:
        raise NotImplementedError()

    rays_d = dirs
    return rays_d

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def get_view_direction(thetas, phis, overhead, front):
    #                   phis: [B,];          thetas: [B,]
    # Default: front = 60, overhead = 30
    # front = 0             [-front/2, front/2) 
    # side (cam left) = 1   [180+front/2, 360-front/2)
    # back = 2              [180-front/2, 180+front/2)
    # side (cam right) = 3  [front/2, 180-front/2)
    # top = 4               [0, overhead]
    # bottom = 5            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 1
    
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 3
    
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


