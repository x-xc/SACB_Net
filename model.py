
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from nn_util import get_act_layer, conv, unfoldNd
from SACB1 import SACB,  cross_Sim
import utils
def tuple_(x, length = 1):
    return x if isinstance(x, tuple) else ((x,) * length)

class Encoder(nn.Module):
    def __init__(self, in_c=1, c=4):
        super(Encoder, self).__init__()
      
        act=("leakyrelu", {"negative_slope": 0.1})  
        norm= 'instance'
        self.conv0 = double_conv(c, 2*c, act=act, norm=norm,
                        pre_fn=conv(in_c,c,act=act,norm=norm))
         
        self.conv1 =  double_conv(2 * c, 4 * c, act=act, norm=norm,
                        pre_fn=nn.AvgPool3d(2))
  
        self.conv2 = double_conv(4 * c, 8 * c, act=act, norm=norm,
                        pre_fn= nn.AvgPool3d(2))

        self.conv3 = double_conv(8 * c, 16* c, act=act, norm=norm,
                        pre_fn=nn.AvgPool3d(2))

        self.conv4 =double_conv(16 * c, 16 * c, act=act, norm=norm,
                        pre_fn=nn.AvgPool3d(2))

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/16

        return out0, out1, out2, out3, out4


def double_conv(in_c, out_c, act, norm='instance', append_fn=None, pre_fn=None):
    layer = nn.Sequential(pre_fn if pre_fn else nn.Identity(),
                          conv(in_c,  out_c, 3,1,1,act=act,norm=norm),
                          conv(out_c, out_c, 3,1,1,act=act,norm=norm),
                          append_fn if append_fn else nn.Identity()
                        )
    return layer

class SACB_Net(nn.Module):
    def __init__(self,
                 inshape=(160,192,160),
                 in_c = 1,
                 ch_scale = 4,
                 num_k = 5, 
                 scale = 1.,
                 mean_type='s'
                ):
        super(SACB_Net, self).__init__()
        self.ch_scale = ch_scale
        self.inshape = inshape
        self.scale = scale
        c = self.ch_scale
        self.mt = mean_type
        if type(num_k) is not tuple:
            self.num_k = tuple_(num_k, length=4)
        else: self.num_k = num_k
        self.encoder = Encoder(in_c=in_c, c=c)
        act=("leakyrelu", {"negative_slope": 0.1}) 
     
        proj_n = 1
        self.up_tri = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.conv1 = double_conv(2*c, 2*c, act=act)
        self.cross_sim = cross_Sim()
        
        self.sacb_proj2 = SACB(4*c,   4*c, in_proj_n=proj_n, ks=3, mean_type=self.mt, num_k=self.num_k[0], act=act, residual=True)
        self.sacb_proj3 = SACB(8*c,   8*c, in_proj_n=proj_n, ks=3, mean_type=self.mt, num_k=self.num_k[1], act=act, residual=True)
        self.sacb_proj4 = SACB(16*c, 16*c, in_proj_n=proj_n, ks=3, mean_type=self.mt, num_k=self.num_k[2], act=act, residual=True)
        self.sacb_proj5 = SACB(16*c, 16*c, in_proj_n=proj_n, ks=3, mean_type=self.mt, num_k=self.num_k[3], act=act, residual=True)
        self.conv1_out = double_conv(2*2*c, 2*c, act=act, append_fn=conv(2*c,3, 3,1,1, act=None))
        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(utils.SpatialTransformer([s // 2**i for s in inshape]))
    
    def set_k(self, k):
        if type(k) is not tuple:
            k = tuple_(k, length=4)
        self.sacb_proj5.set_num_k(k[0])
        self.sacb_proj4.set_num_k(k[1])
        self.sacb_proj3.set_num_k(k[2])
        self.sacb_proj2.set_num_k(k[3])
        
    def forward(self, x, y, softsign_last=False):
        # encode stage
        M1, M2, M3, M4, M5 = self.encoder(x)
        F1, F2, F3, F4, F5 = self.encoder(y)

        F5, M5 = self.sacb_proj5(F5), self.sacb_proj5(M5)
        phi_5 = self.cross_sim(M5, F5)
        phi_5 = self.up_tri(2.* phi_5)

        M4 = self.transformer[3](M4, phi_5)
        F4, M4 = self.sacb_proj4(F4), self.sacb_proj4(M4)
        delta_phi_4 = self.cross_sim(M4, F4)
        phi_4 = self.up_tri(2.* (self.transformer[3](phi_5, delta_phi_4) + delta_phi_4))
            
        M3 = self.transformer[2](M3, phi_4)
        F3, M3 = self.sacb_proj3(F3), self.sacb_proj3(M3)
        delta_phi_3 = self.cross_sim(M3, F3)
        phi_3 = self.up_tri(2.* (self.transformer[2](phi_4, delta_phi_3) + delta_phi_3))

        M2 = self.transformer[1](M2, phi_3)
        F2, M2 = self.sacb_proj2(F2), self.sacb_proj2(M2)
        delta_phi_2 = self.cross_sim(M2, F2)
        phi_2 = self.up_tri(2.* (self.transformer[1](phi_3, delta_phi_2) + delta_phi_2))

        M1 = self.transformer[0](M1, phi_2)
        F1, M1 = self.conv1(F1), self.conv1(M1)
        delta_phi_1 = self.conv1_out(torch.cat([F1, M1],1))
        if softsign_last:
            delta_phi_1 = F.softsign(delta_phi_1)
        # w = self.conv1_out(torch.cat([M1,F1],1))
        Phi = self.transformer[0](phi_2, delta_phi_1) + delta_phi_1
        
        x_warped = self.transformer[0](x, Phi)
        return x_warped, Phi
        
