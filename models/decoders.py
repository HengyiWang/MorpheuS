'''
The code here is mostly adapted from NDR
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True,
                 geo_init=False, inside_outside=False, geo_bias=0.5, weight_norm=True, bias_init=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []

        for l in range(num_layers):
            dim_in = self.dim_in if l == 0 else self.dim_hidden
            dim_out = self.dim_out if l == num_layers - 1 else self.dim_hidden
            lin = nn.Linear(dim_in, dim_out, bias=bias)
            
            if geo_init:
                # Last layer init
                if l == self.num_layers - 1:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dim_in), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -geo_bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dim_in), std=0.0001)
                        torch.nn.init.constant_(lin.bias, geo_bias)
                
                elif l == 0:
                    # Keep only the first 3 dimensions for position
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim_out))
                
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))
                
            if bias_init is not None:
                if l == self.num_layers - 1:
                    print('bias init')
                    torch.nn.init.constant_(lin.bias, -bias_init)
                
            
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            
            net.append(lin)

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class DMLP(nn.Module):
    '''
    MLP initialization for deformation network. 
    Not used in the final version. Keep it here 
    for potential future use.
    '''
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, 
                 bias=True, weight_norm=True, pos=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            dim_in = self.dim_in if l == 0 else self.dim_hidden
            dim_out = self.dim_out if l == num_layers - 1 else self.dim_hidden
            lin = nn.Linear(dim_in, dim_out, bias=bias)
            
            
            if l == num_layers - 1:
                # torch.nn.init.constant_(lin.bias, 0.0)
                # torch.nn.init.constant_(lin.weight, 0.0)
                pass
            elif l == 0:
                # torch.nn.init.constant_(lin.bias, 0.0)
                # torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim_out))
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.constant_(lin.bias[3:], 0.0)
            
            else:
                pass
                # torch.nn.init.constant_(lin.bias, 0.0)
                # torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))
                
            
            if weight_norm and l != num_layers - 1:
                lin = nn.utils.weight_norm(lin)
            
            
            net.append(lin)

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class TMLP(nn.Module):
    '''
    MLP initialization for topology network. 
    Not used in the final version. Keep it here 
    for potential future use.
    '''
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, 
                 bias=True, weight_norm=True, pos=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        
        for l in range(num_layers):
            dim_in = self.dim_in if l == 0 else self.dim_hidden
            dim_out = self.dim_out if l == num_layers - 1 else self.dim_hidden
            lin = nn.Linear(dim_in, dim_out, bias=bias)
            
            
            if l == num_layers - 1:
                # torch.nn.init.normal_(lin.weight, mean=0.0, std=1e-5)
                # torch.nn.init.constant_(lin.bias, 0.0)
                pass
            elif l == 0:
                # torch.nn.init.constant_(lin.bias, 0.0)
                # torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim_out))
                torch.nn.init.constant(lin.weight[:, 3:], 0.0)
                torch.nn.init.constant_(lin.bias[3:], 0.0)
                
            
            else:
                pass
                # torch.nn.init.constant_(lin.bias, 0.0)
                # torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))  
            
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            
            net.append(lin)

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x
    