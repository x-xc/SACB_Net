import torch
import torch.nn as nn 
import numpy as np
from torch.nn.modules.utils import _pair, _single, _triple
import torch.nn.functional as F
from torch.distributions.normal import Normal
# from util import default_, tuple_
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Norm, Act
from monai.networks.blocks import UpSample
from monai.utils import InterpolateMode, UpsampleMode
from monai.networks.layers.utils import get_act_layer

Act.add_factory_callable("softsign", lambda: nn.modules.Softsign)

def exists(x): return x is not None 

def _make_weight(in_channels, kernel_size, device, dtype):
    kernel_size_numel =  int(np.prod((kernel_size)))
    repeat = [in_channels, 1] + [1 for _ in kernel_size]

    return (
        torch.eye(kernel_size_numel, device=device, dtype=dtype)
        .reshape((kernel_size_numel, 1, *kernel_size))
        .repeat(*repeat)
    )
          
def unfoldNd(input, kernel_size, dilation=1, padding=0, stride=1):
    b,c  = input.shape[0], input.shape[1]
    # get convolution operation
    spatial_dim = input.dim() - 2
    conv = [F.conv1d, F.conv2d, F.conv3d][spatial_dim-1]
    _tuple = [_pair, _single, _triple]
    
    kernel_size =_tuple[spatial_dim-1](kernel_size)
    # prepare one-hot convolution kernel
    weight = _make_weight(c, kernel_size, input.device, input.dtype)

    unfold = conv(
        input,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=c,
    )

    return unfold


class STN(nn.Module):
    def __init__(self,  device='cuda', norm=True):
        super(STN, self).__init__()
        self.dev = device
        self.norm = norm
    
    def reference_grid(self, shape):
        dhw = shape[2:]
        if self.norm: 
            grid = torch.stack(torch.meshgrid([torch.linspace(-1, 1, s) for s in dhw], indexing='ij'), dim=0)
        else: 
            grid = torch.stack(torch.meshgrid([torch.arange(0, s) for s in dhw], indexing='ij'), dim=0)
        grid = nn.Parameter(grid.float(), requires_grad=False).to(self.dev)
        return grid
    
    def forward(self, image, ddf, mode='bilinear', p_mode='zeros'):
        spatial_dims = len(image.shape) - 2
        grid =  ddf + self.reference_grid(image.shape)
        if not self.norm:
            for i in range(len(spatial_dims)):
                grid[:, i, ...] = 2 * (grid[:, i, ...] / (spatial_dims[i] - 1) - 0.5)
        grid = grid.movedim(1, -1)
        idx_order = list(range(spatial_dims - 1, -1, -1))
        grid = grid[..., idx_order]  # z, y, x -> x, y, z
        warped = F.grid_sample(image, grid, mode=mode, padding_mode=p_mode, align_corners=True)
        return grid, warped
    
class svf_exp(nn.Module):
    def __init__(self, time_step=7, device ='cuda'):
        super(svf_exp,self).__init__()
        self.time_step = time_step
        self.warp = STN(device=device)
    def forward(self, flow):
        flow = flow / (2 ** self.time_step)
        for _ in range(self.time_step):
            flow = flow + self.warp(image=flow, ddf=flow,  mode='bilinear', p_mode="border")[1]
        return flow

class flow_out(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3):
        conv3d = nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=k//2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)
        
class flow_out2(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3):
        conv3d = nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=k//2, bias=False)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        # conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        act = nn.Softsign()
        super().__init__(conv3d, act)
        
def conv(in_c, out_c, k=3, s=1, p=1, dim=3, bias=True, 
         act='prelu', norm=None, drop=None):
    return Convolution(
            spatial_dims=dim,
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            strides=s,
            padding=p,
            adn_ordering="NA" if exists(norm) else "A",
            act=act,
            dropout=drop,
            norm=norm,
            bias=bias,
        )

def up_sample(in_c, out_c, dim=3, train=True, act='prelu', s=2, align=True, mode='nearest'):
    if not train: act = None
    if mode =='nearest': align = None
#   mode = ['nearest', InterpolateMode.BILINEAR, InterpolateMode.TRILINEAR][dim-1]
    return nn.Sequential(UpSample(spatial_dims=dim, in_channels=in_c, out_channels=out_c,
                                scale_factor=s, kernel_size=2, size=None,
                                mode=UpsampleMode.DECONV if train else UpsampleMode.NONTRAINABLE,
                                pre_conv='default', 
                                interp_mode=mode, align_corners=align,
                                bias=True,
                                apply_pad_pool=True),
                                get_act_layer(act) if exists(act) else nn.Identity())


class conv_twice(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3, s=1, act='prelu', norm=None):
        super(conv_twice, self).__init__()
        self.conv1 = conv( in_ch, out_ch, 3, s, 1, dim=dim, act=act, norm=norm)
        self.conv2 = conv(out_ch, out_ch, 3, 1, 1, dim=dim, act=act, norm=norm)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x    

class LayerNorm(torch.nn.Module):
    def __init__(self, dim):
        
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        d,h,w = x.shape[-3:]
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.norm(x)
        return rearrange(x, 'b (d h w) c -> b c d h w', d=d,h=h,w=w)    
    
class conv_twice_LN(nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super(conv_twice_LN, self).__init__()
        self.conv1 = nn.Conv3d( in_ch, out_ch, 3, s, 1)
        self.ln1 = LayerNorm(out_ch)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1)
        self.ln2 = LayerNorm(out_ch)
        self.act2 = nn.PReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.act2(x)
        return x    
 
 
       
class Up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3, 
                skip_c=0, act='prelu', norm=None, 
                train=True, mode='nearest', align=True):
        super(Up_conv, self).__init__()
        self.skip_c = skip_c
        self.up = up_sample(in_ch, in_ch, dim=dim, train=train, act=act, mode=mode, align=align)
        self.conv1 = conv(in_ch + skip_c, out_ch, 3, 1, 1, dim=dim, act=act, norm=norm)
        self.conv2 = conv(out_ch, out_ch, 3, 1, 1, dim=dim, act=act, norm=norm)
    
    def forward(self, x, skip_in=None):
        x = torch.cat((self.up(x), skip_in), 1) if self.skip_c>0 else self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
