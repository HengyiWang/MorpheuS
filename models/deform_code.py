import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCode(nn.Module):
    '''
    Multi-resolution 1D feature grid for deformation field.
    '''
    def __init__(self, sizes, c):
        super().__init__()
        
        
        self.volumes = nn.ParameterList()
        for size in sizes:
            self.volumes.append(nn.Parameter(torch.randn((1, c, size, 1))))
        
        self.activation = nn.Softplus() # Not used it
            
    
    def sample(self, t):
        # t (n, 1)
        
        t = torch.clamp(t, 0, 1)
        t = t * 2 -1
        t = t[None, :, None, :]
        t = torch.cat([torch.zeros_like(t), t], dim=-1)
        
        feat = []
        
        for volume in self.volumes:
            sample_feat = F.grid_sample(volume, t, align_corners=True).squeeze()
            if len(sample_feat.shape) == 1:
                sample_feat = sample_feat.unsqueeze(1)
            feat.append(sample_feat.permute(1, 0))
            

        feat = torch.cat(feat, dim=-1)
        
        # feat = self.activation(feat)
        return feat

    def get_code(self, level=-1):
        return self.volumes[level].squeeze().permute(1, 0)