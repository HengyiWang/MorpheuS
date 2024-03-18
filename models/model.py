import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

from .decoders import MLP
from .pose import PoseArray
from .encodings import get_encoder
from .deform_code import MultiCode
from .density import LaplaceDensity
from utils import safe_normalize


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(max=15))

trunc_exp = _trunc_exp.apply

class scene_representation(nn.Module):
    def __init__(self, 
                 config,
                 bound,
                 max_level=None,
                 num_layers=3,
                 num_layers_t=6,
                 hidden_dim=64,
                 hidden_dim_t=128, 
                 hidden_dim_tpo=128,
                 num_layers_bg=2,
                 geo_dim=32,
                 deform_dim=16,
                 hidden_dim_bg=32,
                 amb_dim=2,
                 num_frames=None,
                 use_app=False,
                 use_t=False,
                 color_grid=True,
                 use_joint=False,
                 encode_topo=False,
                 encode_deform=True
                 ):
        """
        Initialize the model.

        Args:
            config: Configuration parameters for the model.
            bound: Bound for scene representation.
            max_level: This is for coarse-to-fine training.
            num_layers (int, optional): Number of layers for canonical field decoder. Defaults to 3.
            num_layers_t (int, optional): Number of layers for deformation field decoder. Defaults to 6.
            hidden_dim (int, optional): Hidden dimension for canonical field decoder. Defaults to 64.
            hidden_dim_t (int, optional): Hidden dimension for deformation net decoder. Defaults to 128.
            hidden_dim_tpo (int, optional): Hidden dimension for topology net decoder. Defaults to 128.
            num_layers_bg (int, optional): Number of layers for background network. Defaults to 2.
            geo_dim (int, optional): Geometric feature dimension. Defaults to 32.
            deform_dim (int, optional): Dimension for deformation code. Defaults to 16.
            hidden_dim_bg (int, optional): Hidden dimension for background network. Defaults to 32.
            amb_dim (int, optional): Ambient dimension (Output of topology network). Defaults to 2.
            num_frames (int, optional): Number of frames.
            use_app (bool, optional): Whether to use appearance code. Defaults to False.
            use_t (bool, optional): Whether to use temporal encoding with deformation code. Defaults to False.
            color_grid (bool, optional): Whether to use color grid encoding. Defaults to True.
            use_joint (bool, optional): Whether to use joint encoding. Defaults to False.
            encode_topo (bool, optional): Whether to encode topological features. Defaults to False.
            encode_deform (bool, optional): Whether to encode deformation field. Defaults to True.
        """
        
        super().__init__()
        
        self.config = config
        self.bound = bound
        self.max_level = max_level
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_dim = geo_dim
        self.num_frames = num_frames
        self.use_t = use_t
        self.use_app = use_app
        self.use_joint = use_joint
        self.encode_topo = encode_topo
        self.encode_deform = encode_deform
        
        self.pose_array = PoseArray(num_frames)
        
        
        # Deformation Field
        
        self.encoder_t, self.in_dim_t = self.get_encodings(encodings='frequency_torch', 
                                                           input_dim=1, 
                                                           multires=6,
                                                           use_encoding=self.use_t,
                                                           output_dim=0)
        
        # Not used in final version
        # NOTE: We find that topological feature output by topology network need
        # to be smooth. Using frequency encoding will ruin the temporal coherence.
        # Recommend to use other smooth encoding if you are interested in it.
        self.encoder_topo, self.in_dim_amb = self.get_encodings(encodings='frequency_torch',
                                                                input_dim=amb_dim, 
                                                                multires=4,
                                                                use_encoding=self.encode_topo,
                                                                output_dim=amb_dim)
        
        self.encoder_deform, self.in_dim_deform = self.get_encodings(encodings='frequency_torch',
                                                                     input_dim=3, 
                                                                     multires=6,
                                                                     use_encoding=self.encode_deform,
                                                                     output_dim=3)
        
        self.deform_code, self.deform_dim = self.get_encodings(encodings='MultiCode',
                                                               deform_dim=deform_dim,
                                                               code_dim=[num_frames//8, num_frames//4, num_frames],
                                                               output_dim=0)
        
        # Not used in final version. Keep it for potential future use
        # NOTE: NDR uses appearance code for color. However, we want
        # the color of the target to be consistent and does not reply
        # on the time. Therefore, we did not use the appearance code
        self.app_code, self.app_dim = self.get_encodings(encodings='MultiCode',
                                                         deform_dim=deform_dim,
                                                         use_encoding=self.use_app,
                                                         code_dim=[num_frames//8, num_frames//4, num_frames],
                                                         output_dim=0)
        
        self.deform_net = MLP(self.in_dim_t+self.in_dim_deform+self.deform_dim, 3, hidden_dim_t, num_layers_t, bias=True)
        self.topo_net = MLP(self.in_dim_t+self.in_dim_deform+self.deform_dim, amb_dim, hidden_dim_tpo, num_layers_t, bias=True)
        
        
        # Hyper-dimensional Canonical Field
        
        self.encoder, self.in_dim = self.get_encodings(encodings='hashgrid',
                                                       input_dim=3,
                                                       num_levels=16,
                                                       log2_hashmap_size=15, 
                                                       desired_resolution=128, 
                                                       interpolation='linear')
        
        self.encoder_c, self.in_dim_c = self.get_encodings(encodings='hashgrid' if color_grid else 'frequency_torch',
                                                           input_dim=3,
                                                           multires=6,
                                                           num_levels=16,
                                                           log2_hashmap_size=15, 
                                                           desired_resolution=128, 
                                                           interpolation='linear')
        
        # This is for geometric initialization. 
        # NOTE: We find that a simple xyz is enough for the geometric initialization. 
        # Other coordinate encoding can be also used (like joint encoding in Co-SLAM, 3QFP),
        # but remember to only keep xyz and mask out the rest part of the coordinate encoding.
        self.encoder_xyz, self.in_dim_xyz = self.get_encodings(encodings='frequency_torch', 
                                                           input_dim=3, 
                                                           multires=6,
                                                           use_encoding=self.use_joint,
                                                           output_dim=3)
        
        self.sdf_net = MLP(self.in_dim+self.in_dim_amb+self.in_dim_xyz,
                           1+geo_dim, hidden_dim, num_layers,
                           bias=True, geo_init=True, geo_bias=0.4, weight_norm=False)
        
        self.color_net = MLP(self.in_dim_c+geo_dim+self.app_dim, 
                             3, hidden_dim, num_layers, bias=True)
        
        # Background network
        if self.config['model']['bg_radius'] > 0:
            self.encoder_bg, self.in_dim_bg = self.get_encodings('frequency_torch', 
                                                                 input_dim=3, 
                                                                 multires=6)
            self.encoder_bg_t, self.in_dim_bg_t = self.get_encodings('frequency_torch', 
                                                                 input_dim=1, 
                                                                 multires=6)
        
            self.bg_net = MLP(self.in_dim_bg + self.in_dim_bg_t, 3, hidden_dim_bg, num_layers_bg, bias=True)
        
        self.density_activation = trunc_exp if self.config['model']['activation'] == 'exp' else biased_softplus
        self.sdf2density = LaplaceDensity({'beta':0.1})
        
        # TODO: Describle what those params mean and how to use it
        # self.sample_param = nn.Parameter(torch.ones(2,))
        # self.sample_novel_r = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(6)])#nn.Parameter(torch.ones(6,2))
        # self.sample_novel_f = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(6)])#nn.Parameter(torch.ones(6,2))
        
    def get_encodings(self, encodings='frequency_torch', 
                      use_encoding=True,
                      input_dim=3,
                      multires=6,
                      num_levels=16,
                      log2_hashmap_size=15, 
                      desired_resolution=128, 
                      interpolation='linear',
                      output_dim=None,
                      code_dim=None,
                      deform_dim=None):
        '''
        Get encodings and output dimensions
        '''
        
        if not use_encoding:
            assert output_dim is not None, 'output dim should not be None'
            return None, output_dim
        
        if encodings == 'MultiCode':
            assert code_dim is not None, 'code_dim should not be None'
            assert deform_dim is not None, 'deform_dim should not be None'
            
            return MultiCode(code_dim, deform_dim), len(code_dim) * deform_dim
            
        
        return get_encoder(encodings, 
                           input_dim=input_dim,
                           multires=multires,
                           num_levels=num_levels,
                           log2_hashmap_size=log2_hashmap_size,
                           desired_resolution=desired_resolution,
                           interpolation=interpolation)
    
    def get_deform_code(self, t, app=False):
        '''
        Get deformation code given timestep
        '''
        if app:
            code = self.app_code.sample(t)
        else:
            code = self.deform_code.sample(t)
        return code
    
    def get_RT(self, frame_ids):
        '''
        Get rotation and translation matrices given frame ids
        '''
        frame_ids = frame_ids.squeeze()
        if self.pose_array is not None:
            R = self.pose_array.get_rotation_matrices(frame_ids)
            T = self.pose_array.get_translations(frame_ids)
        else:
            R, T = None, None

        return R, T
    
    def get_topo(self, x, t):
        '''
        Get ambient coordinate (i.e. topology) given input position and timestep
        '''
        
        x_enc = self.encoder_deform(x, max_level=self.max_level)
        code = self.get_deform_code(t)
        
        if self.use_t:
            # Use temporal frequency encoding + deform code
            t_enc = self.encoder_t(t, max_level=self.max_level)
            topo = self.topo_net(torch.cat([x_enc, t_enc, code], dim=-1))
        else:
            # Use deform code only
            topo = self.topo_net(torch.cat([x_enc, code], dim=-1))
        
        if self.encode_topo:
            topo = self.encoder_topo(topo, max_level=self.max_level)
        
        return topo
    
    def get_sigma_albedo(self, x, topo=None, app_code=None, return_color=True):
        '''
        get sdf, sigma and albedo
        '''
        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)
        
        if topo is None:
            topo = torch.zeros((x.shape[0], self.in_dim_amb), device=x.device)
        
        if app_code is None and self.use_app:
            app_code = torch.zeros((x.shape[0], self.deform_dim), device=x.device)
        
        if self.use_joint:
            enc_xyz = self.encoder_xyz(x, bound=self.bound, max_level=self.max_level)
            sdf_feat = torch.cat([enc_xyz, enc, topo], dim=-1)
        else:
            sdf_feat = torch.cat([x, enc, topo], dim=-1)
        
        h = self.sdf_net(sdf_feat)
        sdf = h[..., 0]
        sigma = self.sdf2density(sdf) 
        
        if return_color:
            enc_color = self.encoder_c(x, bound=self.bound, max_level=self.max_level)
            color_feat = torch.cat([enc_color, h[..., 1:]], dim=-1)
            
            if self.use_app:
                color_feat = torch.cat([color_feat, app_code], dim=-1)
            
            albedo = torch.sigmoid(self.color_net(color_feat))
        
        else:
            albedo = None
        
        return sdf, sigma, albedo 
    
    def get_params_all(self, lr):
        '''
        Get all parameters for the model
        '''
        params = [
            {'name':'encoder_sdf', 'params': self.encoder.parameters(), 'lr': lr},
            {'name':'encoder_color', 'params': self.encoder_c.parameters(), 'lr': lr},
            {'name':'decoder_sdf', 'params': self.sdf_net.parameters(), 'lr': lr},
            {'name':'decoder_topo', 'params': self.topo_net.parameters(), 'lr': lr},
            {'name':'decoder_color', 'params': self.color_net.parameters(), 'lr': lr},
            {'name':'density', 'params': self.sdf2density.parameters(), 'lr': lr/2.},
            #{'name':'intri', 'params': self.sample_param, 'lr': lr/10.},
            {'name':'decoder_deform', 'params': self.deform_net.parameters(), 'lr': lr},
            {'name':'code_deform', 'params': self.deform_code.parameters(), 'lr': lr},
            {'name':'pose', 'params': self.pose_array.parameters(), 'lr': lr/10.},
        ]       

        if self.config['model']['bg_radius'] > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'name': 'decoder_bg', 'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.use_app:
            params.append({'name':'code_app', 'params': self.app_code.parameters(), 'lr': lr}),
        
        return params
    
    def pose_optimisation(self, rays_o, rays_d, frame_ids):
        '''
        Apply pose correction module to the input rays
        '''
        frame_ids = frame_ids.squeeze()
        if self.pose_array is not None:
            R = self.pose_array.get_rotation_matrices(frame_ids)
            t = self.pose_array.get_translations(frame_ids)
            rays_o = rays_o + t
            rays_d = torch.sum(rays_d[..., None, :] * R, -1)

        return rays_o, rays_d
    
    def volume_rendering(self, z_vals, sdf):
        '''
        VolSDF volume rendering, did not use this part in final version
        '''
        density_flat = self.sdf2density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights
    
    def finite_difference_normal(self, x, epsilon=2e-3, topo=None):
        '''
        Caculate normal using finite difference (numerical gradient)
        '''
        # x: [N, 3]
        dx_pos, _, _ = self.get_sigma_albedo((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound), topo=topo, return_color=False)
        dx_neg, _, _ = self.get_sigma_albedo((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound), topo=topo, return_color=False)
        dy_pos, _, _ = self.get_sigma_albedo((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound), topo=topo, return_color=False)
        dy_neg, _, _ = self.get_sigma_albedo((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound), topo=topo, return_color=False)
        dz_pos, _, _ = self.get_sigma_albedo((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound), topo=topo, return_color=False)
        dz_neg, _, _ = self.get_sigma_albedo((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound), topo=topo, return_color=False)
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return normal
    
    def normal(self, x, t=None, cano=False, topo=None):
        '''
        Get normal given poisitions in observation space and timesteps
        '''
        if t is not None and not cano:
            deform, topo, app_code = self.warp(x, t)
            x = x + deform
        
        normal_raw = self.finite_difference_normal(x, topo=topo)
        normal = safe_normalize(normal_raw)
        normal = torch.nan_to_num(normal)
        return normal, normal_raw
    
    def background(self, d, t):
        '''
        Background color
        '''
        
        h = self.encoder_bg(d) # [N, C]
        h_t = self.encoder_bg_t(t, max_level=self.max_level)
         
        color = torch.sigmoid(self.bg_net(torch.cat([h, h_t], dim=-1)))
         
        return color
  
    def warp(self, x, t):
        '''
        Apply deformation field to the input position and timestep
        '''
        code = self.get_deform_code(t)
        
        app_code = None
        if self.use_app:
            app_code = self.get_deform_code(t, app=True)
        
        x_enc = self.encoder_deform(x, max_level=self.max_level)
        
        if self.use_t:
            # Use temporal frequency encoding + deform code
            t_enc = self.encoder_t(t, max_level=self.max_level)
            deform = self.deform_net(torch.cat([x_enc, t_enc, code], dim=-1))
            topo = self.topo_net(torch.cat([x_enc, t_enc, code], dim=-1))
        else:
            # Use deform code only
            deform = self.deform_net(torch.cat([x_enc, code], dim=-1))
            topo = self.topo_net(torch.cat([x_enc, code], dim=-1))
        
        if self.encode_topo:
            topo = self.encoder_topo(topo, max_level=self.max_level)
        
        return deform, topo, app_code
            
    def density(self, x, t=None, cano=False, allow_shape=False, return_color=True):
        """
        Compute the density of the model at the given points.

        Args:
            x: Input positions.
            t : Timestep If None, directly query the canonical field without defomation field.
            cano: Flag indicating whether to directly use the canonical field. 
            allow_shape: Flag indicating whether to allow shape inconsistency between x and t. If False and x and t have different shapes, an exception is raised. Default is False.
            return_color: Flag indicating whether to return the color information along with the density. Default is True.



        """
        if cano or t is None:
            # NOTE: Here topo = None will assign all zero vectors for topo
            # That is to say, we take zero topology as canonical model
            topo = None 
            app_code = None
        
        else:
            # Warp the input point to the cano space, and query cano field
            if isinstance(t, float):
                t = t * torch.ones(x.shape[0], 1, device=x.device)
            
            if t is not None and x.shape[0] == t.shape[0]:
                deform, topo, app_code = self.warp(x, t)
                x = x + deform
            
            elif t is not None and x.shape[0] != t.shape[0]:
                if not allow_shape:
                    raise Exception('Shape inconsistent!!!')
                
                deform, topo, app_code = self.warp(x, t[0, 0] * torch.ones(x.shape[0], 1, device=x.device))
                x = x + deform
                        
        sdf, sigma, albedo = self.get_sigma_albedo(x, topo=topo, app_code=app_code, return_color=return_color)
        
        return {
            'sdf': sdf,
            'sigma': sigma,
            'albedo': albedo,
        } 
          
    def forward(self, x, t, light_dir=None, ratio=1, 
                shading='albedo', cano=False, return_color=True):
        """
        Forward pass of the model.

        Args:
            x: Input position.
            t: timestep.
            light_dir: Light direction tensor. Defaults to None.
            ratio: Ratio for shading. Defaults to 1.
            shading: Shading type.
            cano: Flag indicating whether query without deformation field
            return_color: Flag indicating whether to return color. Defaults to True.
            
        """
        if cano:
            x_ori = x
            deform = None
            topo = None
            app_code = None
        else:
            deform, topo, app_code = self.warp(x, t)
            x_ori = x + deform
        
        sdf, sigma, albedo = self.get_sigma_albedo(x_ori, topo, app_code, return_color)
        
        
        if shading == 'albedo':
            normal = None
            color = albedo
            noraml_raw = None
        
        else:
            # NOTE: This part we skip the deformation network to obtain normal
            # for canonical space regularization. We find this normal is still 
            # overall good enough for calculating the lambertian shading. Thus, 
            # we use it directly to save computational cost.
            
            normal, noraml_raw = self.normal(x, topo=topo)
            lambertian = ratio + (1 - ratio) * (normal * light_dir).sum(-1).clamp(min=0) # [N,]
            
            if shading == 'textureless':
                # Nothing to do with albedo, only train normal branch
                color = lambertian.unsqueeze(-1).repeat(1, 3) 
            elif shading == 'normal':
                color = (normal + 1) / 2
            
            else: # lambertian
                color = albedo * lambertian.unsqueeze(-1)
        
        return sdf, sigma, color, normal, deform, noraml_raw
            
        
        
        
