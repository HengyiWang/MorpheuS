import os
import cv2
import torch
import imageio
import trimesh
import pyrender

import numpy as np
import open3d as o3d


from tqdm import tqdm
from copy import deepcopy
from scipy.spatial import cKDTree as KDTree


def cull_from_one_pose(points, pose, K, H, W, rendered_depth, eps=0.005, depth_gt=None, remove_missing_depth=True):

    # OpenGL pose
    c2w = deepcopy(pose)
    # to OpenCV
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]

    # pts under camera frame
    camera_space = rotation @ points.transpose() + translation[:, None]  # [3, N]
    uvz = (K @ camera_space).transpose()  # [N, 3]
    pz = uvz[:, 2] + 1e-8
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # step 1: inside frustum
    in_frustum_mask = (0 <= px) & (px <= W - 1) & (0 <= py) & (py <= H - 1) & (pz > 0)
    u = np.clip(px, 0, W - 1).astype(np.int32)
    v = np.clip(py, 0, H - 1).astype(np.int32)

    # step 2: not occluded
    obs_mask = in_frustum_mask & (pz < (rendered_depth[v, u] + eps))

    # step 3: valid depth in gt
    if remove_missing_depth:
        invalid_mask = in_frustum_mask & (depth_gt[v, u] <= 0.)
    else:
        invalid_mask = np.zeros_like(in_frustum_mask)

    return obs_mask, invalid_mask

def render_depth_maps(mesh, poses, K, H, W, yfov=60.0, far=2.0):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    # camera = pyrender.PerspectiveCamera(yfov=math.radians(yfov), aspectRatio=W/H, znear=0.01, zfar=far)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

    depth_maps = []
    for pose in poses:
        scene.set_pose(camera_node, pose)
        depth = renderer.render(scene, render_flags)
        depth_maps.append(depth)

    return depth_maps

def render_depth_maps_doublesided(mesh, poses, K, H, W, far=100.0):
    # depth_maps_1 = render_depth_maps(mesh, poses, H, W, yfov, far)
    depth_maps_1 = render_depth_maps(mesh, poses, K, H, W, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    # depth_maps_2 = render_depth_maps(mesh, poses, H, W, yfov, far)
    depth_maps_2 = render_depth_maps(mesh, poses, K, H, W, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]  # it's a pass by reference, so I restore the original order
    depth_maps = []
    for i in range(len(depth_maps_1)):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where((depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map), depth_maps_2[i], depth_map)
        depth_maps.append(depth_map)

    return depth_maps

def cull_one_mesh(dataset, mesh_path, save_path, c2w, depth_gt,
                  remove_missing_depth=True, remove_occlusion=True, eps=0.005,
                  scene_bounds=None, subdivide=True, max_edge=0.01, platform='egl'):
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = mesh.vertices  # [V, 3]
    triangles = mesh.faces  # [F, 3]
    colors = mesh.visual.vertex_colors  # [V, 3]

    if subdivide:
        vertices, triangles = trimesh.remesh.subdivide_to_size(vertices, triangles, max_edge=max_edge, max_iter=10)

    os.environ['PYOPENGL_PLATFORM'] = platform

    # only one pose a time
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.cpu().numpy()

    K, H, W = dataset.intrinsics.cpu().numpy(), dataset.H, dataset.W
    rendered_depth_map = render_depth_maps_doublesided(mesh, [c2w], K, H, W, far=10.0)[0]

    # we don't need subdivided mesh to render depth
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()

    # Cull faces
    points = vertices[:, :3]  # [V, 3]
    obs_mask, invalid_mask = cull_from_one_pose(points, c2w, K, H, W,
                                                rendered_depth=rendered_depth_map,
                                                depth_gt=depth_gt,
                                                remove_missing_depth=remove_missing_depth,
                                                eps=eps)
    obs1 = obs_mask[triangles[:, 0]]
    obs2 = obs_mask[triangles[:, 1]]
    obs3 = obs_mask[triangles[:, 2]]
    obs_mask = obs1 | obs2 | obs3
    inv1 = invalid_mask[triangles[:, 0]]
    inv2 = invalid_mask[triangles[:, 1]]
    inv3 = invalid_mask[triangles[:, 2]]
    invalid_mask = inv1 & inv2 & inv3
    valid_mask = obs_mask & (~invalid_mask)
    triangles_observed = triangles[valid_mask, :]

    # culled mesh
    mesh = trimesh.Trimesh(vertices, triangles_observed, vertex_colors=colors, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.export(save_path)
    
def cull_meshes(mesh_dir, save_dir, dataset, target):
    """
    :param mesh_dir: dir to reconstructed meshes
    :return:
    """
    
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(dataset.num_frames)):
        data = dataset.sample_real_view_rays(i)
        c2w, depth_gt = dataset.poses[i], data["depth"].squeeze().numpy()
        mesh_path = os.path.join(mesh_dir, "{}_{:04d}.ply".format(target, i))
        save_mesh_path = os.path.join(save_dir, "{}_{:04d}.ply".format(target, i))
        cull_one_mesh(dataset, mesh_path, save_mesh_path, c2w, depth_gt=depth_gt, eps=0.005)

def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    # TODO: use o3d.registration for open3d 0.9.0
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # reg_p2p = o3d.registration.registration_icp(
    #     o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
    #     o3d.registration.TransformationEstimationPointToPoint())
    transformation = reg_p2p.transformation
    return transformation

def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def calc_3d_metric(rec_meshfile, gt_meshfile, align=True, num_points=50000):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, num_points)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, num_points)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    # print('accuracy: ', accuracy_rec)
    # print('completion: ', completion_rec)
    # print('completion ratio: ', completion_ratio_rec)

    return {
        'acc': accuracy_rec,
        'comp': completion_rec,
        'comp ratio': completion_ratio_rec
    }

def eval_mesh_3d(rec_files_list, gt_files_list, save_file, epoch):
    """
    3D metric between culled reconstructed mesh and gt depth mesh
    :return:
    """
    assert len(rec_files_list) == len(gt_files_list), "Length mismatch!!!"
    acc, comp = [], []
    for rec_file, gt_file in zip(rec_files_list, gt_files_list):
        rst_dict = calc_3d_metric(rec_file, gt_file)
        acc.append(rst_dict["acc"])
        comp.append(rst_dict["comp"])
    
    print("Ep_{}:\t Acc:{}\t Comp:{}".format(epoch, np.array(acc).mean(), np.array(comp).mean()), file=open(save_file, "a"))

def eval_depthL1(depth_dir, dataset):
    error_images_dir = os.path.join(os.path.dirname(depth_dir), "depth_error")
    os.makedirs(error_images_dir, exist_ok=True)
    errors = []
    depth_preds_dict = np.load(os.path.join(depth_dir, "depths.npz"))
    
    for i in tqdm(range(dataset.num_frames)):
        depth_pred = depth_preds_dict["depth_{}".format(i)]
        depth_gt = dataset.depths[i]
        if isinstance(depth_gt, torch.Tensor):
            depth_gt = depth_gt.cpu().numpy()
        mask_gt = dataset.masks[i][..., 0] > 0.
        valid_mask = (depth_gt > 0.) & mask_gt
        error_plot = np.abs(depth_gt - depth_pred)
        error_plot[~valid_mask] = 0.
        error_plot[error_plot > 1.0] = 0.
        errors.append(error_plot[error_plot > 0.].mean())
        error_plot = 255. - np.clip(error_plot/error_plot.max(), a_max=1, a_min=0) * 255.
        error_plot = np.uint8(error_plot)
        imageio.imwrite(os.path.join(error_images_dir, "{:04d}.png".format(i)), cv2.applyColorMap(error_plot, cv2.COLORMAP_JET))
    errors = np.array(errors)
    np.savetxt(os.path.join(error_images_dir, "depthL1_scores.txt"), errors, fmt="%.5f")
    np.savetxt(os.path.join(error_images_dir, "depthL1_score_mean.txt"), np.array([errors.mean()]), fmt="%.5f")
    print(errors)

def eval_mesh(workspace, mesh_dir, dataset, target, epoch):
    cull_mesh_dir = os.path.join(workspace, 'mesh_all_culled')
    cull_meshes(mesh_dir, cull_mesh_dir, dataset, target)
    
    gt_files_list = [os.path.join(dataset.data_dir, "mesh/backproj_{}.ply".format(i)) for i in range(dataset.num_frames)]
    rec_files_list = [os.path.join(cull_mesh_dir, "{}_{:04d}.ply".format(target, i)) for i in range(dataset.num_frames)]
    eval_mesh_3d(rec_files_list, gt_files_list, os.path.join(workspace, "metric_3d.txt"), epoch)
    
    for cull_file in rec_files_list:
        try:
            os.remove(cull_file)
            #print(f"Deleted: {cull_file}")
        except OSError as e:
            print(f"Error deleting {cull_file}: {e}")