import torch
import torch.nn as nn

class PoseArray(nn.Module):
    """
    Per-frame camera pose correction.
    The pose correction contains 6 parameters for each pose (3 for rotation, 3 for translation).
    The rotation parameters define Euler angles which can be converted into a rotation matrix.
    """

    def __init__(self, num_frames):
        super(PoseArray, self).__init__()

        self.num_frames = num_frames
        self.num_params = 6

        self.data = nn.Parameter(
            torch.zeros([self.num_frames, self.num_params], dtype=torch.float32)
        )

    def forward(self, ids):
        return self.data[ids]

    def get_translations(self, ids):
        trans = self.data[:, 3:6][ids]
        
        if len(trans.shape) == 1:
            trans = trans[None, ...]

        return trans
    
    def get_rotations(self, ids):
        return self.data[:, 0:3][ids]

    def get_rotation_matrices(self, ids):
        rotations = self.get_rotations(ids)  # [N_frames, 3]
        
        if len(rotations.shape) == 1:
            rotations = rotations[None, ...]

        cos_alpha = torch.cos(rotations[:, 0])
        cos_beta = torch.cos(rotations[:, 1])
        cos_gamma = torch.cos(rotations[:, 2])
        sin_alpha = torch.sin(rotations[:, 0])
        sin_beta = torch.sin(rotations[:, 1])
        sin_gamma = torch.sin(rotations[:, 2])

        col1 = torch.stack([cos_alpha * cos_beta,
                         sin_alpha * cos_beta,
                         -sin_beta], -1)
        col2 = torch.stack([cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma,
                         sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma,
                         cos_beta * sin_gamma], -1)
        col3 = torch.stack([cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma,
                         sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma,
                         cos_beta * cos_gamma], -1)

        return torch.stack([col1, col2, col3], -1)

    def transform_points(self, points, ids):
        R = self.get_rotation_matrices(ids)
        t = self.get_translations(ids)

        return torch.reduce_sum(points[..., None, :] * R, -1) + t
