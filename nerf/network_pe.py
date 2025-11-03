import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs) 
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)    

class PEMLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, N_freqs=10):
        super().__init__()
        self.enconding = PositionalEncoding(in_channels=in_features, N_freqs=N_freqs)
        
        self.net = []
        self.net.append(nn.Linear(self.enconding.out_channels, hidden_features))
        self.net.append(nn.ReLU(True))

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(True))

        final_linear = nn.Linear(hidden_features, out_features)                
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(self.enconding(coords))
        return output

class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        encoding="hashgrid",
        encoding_dir="sphere_harmonics",
        encoding_bg="hashgrid",
        num_layers=4, 
        hidden_dim=256,
        geo_feat_dim=256,
        num_layers_color=4,
        hidden_dim_color=256,
        num_layers_bg=2,
        hidden_dim_bg=64,
        bound=1,

        N_freqs=10,
        fw0=30,
        hw0=30,
        fbs=None,
        alpha_mul=10,
        **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        ## sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        
        # no enconding
        self.encoder, self.in_dim = get_encoder(encoding='None')

        self.enconding_sigma = PositionalEncoding(in_channels=self.in_dim, N_freqs=N_freqs)
        sigma_net = []
        for l in range(num_layers):
            if l == 0:                  # first layer
                sigma_net.append(nn.Linear(self.enconding_sigma.out_channels, hidden_dim))
                sigma_net.append(nn.ReLU(True))
            elif l == num_layers - 1:   # final layer
                sigma_net.append(nn.Linear(hidden_dim, 1 + self.geo_feat_dim))
            else:                       # hidden layers
                sigma_net.append(nn.Linear(hidden_dim, hidden_dim))
                sigma_net.append(nn.ReLU(True))
        
        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding='None')
        
        self.enconding_color = PositionalEncoding(in_channels=self.in_dim_dir + geo_feat_dim, N_freqs=N_freqs)
        color_net =  []            
        for l in range(num_layers_color):
            if l == 0:                  # first layer
                color_net.append(nn.Linear(self.enconding_color.out_channels, hidden_dim_color))
                color_net.append(nn.ReLU(True))
            elif l == num_layers_color - 1:   # final layer
                color_net.append(nn.Linear(hidden_dim_color, 3)) # rgb
            else:                       # hidden layers
                color_net.append(nn.Linear(hidden_dim_color, hidden_dim_color))
                color_net.append(nn.ReLU(True))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)
        x = self.enconding_sigma(x)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        d = self.enconding_color(d)

        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        x = self.enconding_sigma(x)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        d = self.enconding_color(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            # {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            # {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
