import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _triple
from nn_util import get_act_layer, conv, unfoldNd
# import functools
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, reduce
import numpy as np
from kmeans_gpu import KMeans

def tuple_(x, length = 1):
    return x if isinstance(x, tuple) else ((x,) * length)


class KM_GPU():
    def __init__(self,num_k=4, rng_seed=0, 
                 tol=1e-9, m_iter=1e9, fix_rng=True,
                 max_neighbors=160*192*224):
        super(KM_GPU, self).__init__()   
        self.seed = rng_seed
        self.fix_rng = fix_rng
        
        self.km = KMeans(
            n_clusters= num_k,
            max_iter= int(m_iter),
 
            tolerance=tol,
            distance='euclidean',
            sub_sampling=None,
            max_neighbors=max_neighbors,
        )
    def set_k(self, k):
        self.km.n_clusters = k
    def get_cluster_map(self, x):
        if self.fix_rng: np.random.seed(self.seed)
        closest, centroid = self.km.fit_predict(x)
        return closest, centroid
        
class cross_Sim(nn.Module): 
    def __init__(self, win_s=3):
        super(cross_Sim, self).__init__()
        self.wins = win_s
        self.win_len = win_s**3
              
    def forward(self, Fx, Fy, wins=None):
        if wins:
            self.wins = wins
            self.win_len = wins**3
        b, c, d, h, w = Fy.shape
      
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [self.wins] * 3]
        grid = torch.stack(torch.meshgrid(vectors), -1).type(torch.FloatTensor)
   
        G = grid.reshape(self.win_len, 3).unsqueeze(0).unsqueeze(0).to(Fx.device)

        Fy = rearrange(Fy, 'b c d h w -> b (d h w) 1 c')
        pd = self.wins // 2  # 1

        Fx = F.pad(Fx,  tuple_(pd, length=6)) 
     
        Fx = Fx.unfold(2, self.wins, 1).unfold(3, self.wins, 1).unfold(4, self.wins, 1)
        Fx = rearrange(Fx, 'b c d h w wd wh ww -> b (d h w) (wd wh ww) c')

        attn = (Fy @ Fx.transpose(-2, -1))
        sim = attn.softmax(dim=-1)
        out = (sim @ G) 
        out = rearrange(out , 'b (d h w) 1 c -> b c d h w', d=d,h=h,w=w)
    
        return out


class SACB(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride=1,
                 in_proj_n=1,
                 padding=1, dilation=1, groups=1,
                 num_k=4, 
                 act='prelu', residual=True, 
                 mean_type = 's',
                 scale_f=1,
                 n_mlp=1,
                 sample_n = 5,
                 m_iter=1e10,
                 tol   =1e-10,
                 fix_rng= False
                 ):
        super(SACB, self).__init__()
        self.ks       = ks
        self.stride   = stride
        self.padding  =  tuple(x for x in reversed(_triple(padding)) for _ in range(2))
        self.dilation = _triple(dilation)
        self.num_k    = num_k
        self.res      = residual
        self.out_ch = out_ch
        in_ch_n = int(in_ch * in_proj_n)
        self.w   = nn.Parameter(torch.Tensor(out_ch, in_ch_n // groups, self.ks**3))
        # self.b   = nn.Parameter(torch.Tensor(num_k, out_ch)) if bias else None
        self.act = get_act_layer(act) if act else None
        self.reset_parameters()
        self.scale_f = scale_f
        self.mean_type = mean_type
        self.km = KM_GPU(num_k=num_k, rng_seed=0, m_iter=m_iter, tol=tol, fix_rng=fix_rng)
        
        inner_dims = 128 * n_mlp
        inner_dims2 = 64 * n_mlp
        
        self.sample_n = sample_n
        if   mean_type =='s': 
            _in_c = in_ch_n
            self._in_c = _in_c
        elif mean_type =='c': _in_c = self.ks**3
        else: _in_c = in_ch + self.ks**3
        
        self.get_kernel = nn.Sequential(
                nn.Linear(_in_c, inner_dims), nn.ReLU(),
                nn.Linear(inner_dims, inner_dims), nn.ReLU(),
                nn.Linear(inner_dims, self.ks**3), nn.Sigmoid()
                )
        
        self.get_bias = nn.Sequential(
                nn.Linear(in_features=_in_c,  out_features=inner_dims2), nn.ReLU(),
                nn.Linear(in_features=inner_dims2, out_features=inner_dims2), nn.ReLU(),
                nn.Linear(in_features=inner_dims2, out_features=out_ch),
                )   
        # self.proj_in  = conv(in_ch, in_ch_n, 3, 1, 1, bias=False)
        self.proj_in  = conv(in_ch, in_ch_n, 3, 1, 1, act=act, norm='instance')
     
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        # if self.b is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        #     if fan_in != 0:
        #         bound = 1 / math.sqrt(fan_in)
        #         nn.init.uniform_(self.b, -bound, bound)
    
    def set_num_k(self, k):
        self.num_k = k
        self.km.set_k(k)    
               
    def scale(self, x, factor, mode='nearest'):
        if mode == 'nearest':
            return F.interpolate(x, scale_factor=factor, mode=mode)  
        else: 
            return F.interpolate(x, scale_factor=factor, mode='trilinear', align_corners=True)  
    
    def feat_mean(self, x, mean_type='s'):
        if   mean_type == 's': x = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) c', 'mean')
        elif mean_type == 'c': x = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) (k1 k2 k3)', 'mean')
        else: 
            xs = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) c', 'mean')
            xc = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) (k1 k2 k3)', 'mean')
            x = torch.cat([xs, xc], -1)
        return x
       
    def forward(self, x):
      
        b,c,d,h,w = x.shape  
        x_in = x
        x = self.proj_in(x) 
        x_pad = F.pad(x, self.padding)
        x = x_pad.unfold(2,self.ks,self.stride).unfold(3,self.ks,self.stride).unfold(4,self.ks,self.stride)
        # x = unfoldNd(x, kernel_size=3, dilation=1, padding=1, stride=1)
       
        x_mean = self.feat_mean(x, self.mean_type)
        cluster_idx, centroid = self.km.get_cluster_map(x_mean.squeeze(0))
        x = rearrange(x,'b c nd nh nw k1 k2 k3 -> b (c k1 k2 k3) (nd nh nw)') 
        out = torch.zeros(b,self.out_ch, d*h*w).cuda()
        for i in range(self.num_k):
            mask = cluster_idx.eq(i)
            cat_ = centroid[i].unsqueeze(0)
            weight = rearrange(self.get_kernel(cat_), 'b k -> b 1 1 k') * self.w
            bias   = rearrange(self.get_bias(cat_),   'b o -> b o 1')
            out = out + (torch.einsum('b i j , b o i -> b o j', x, rearrange(weight, 'b o i k -> b o (i k)')) + bias) * rearrange(mask, 's -> 1 1 s').float()
           
        out = rearrange(out, 'b o (d h w) -> b o d h w', d=d, h=h, w=w)
        if self.act: out = self.act(out)
        if self.res: out = out + x_in
     
        return out 

