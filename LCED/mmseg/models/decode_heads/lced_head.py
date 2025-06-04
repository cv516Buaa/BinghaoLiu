# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList
import torch.nn.functional as F
import math
import os
import cv2
import numpy as np
from einops import rearrange, repeat, einsum

class ICG(BaseModule): # iterative components generation
    def __init__(self,
                 mem_length, # N: the length of memory for save how many samples' components vector
                 G, # G: number of components
                 smooth_kernel, # s: kernel size of avgpool for smoothness
                 iter_train, # iterations for training
                 r_func, # relevancy function 
                 channels,
                 save_path,
                 version,
                 alpha,
                 ):
        super().__init__()

        self.N = mem_length
        self.G = G
        self.s = smooth_kernel
        self.iter = iter_train
        self.r_func = r_func
        self.channels = channels
        self.save_path = save_path
        self.version = version
        self.load = False
        self.save = False

        self.F_seq = torch.zeros(self.N, self.G, self.channels, requires_grad=False)
        self.idx = 0
        self.G_num = 0
        self.iteration_num = 0
        self.alpha = alpha
        self.save_id = 1

        self.vis_id = 0

        self.avgpool = nn.AvgPool2d(self.s, 1, int((self.s - 1) / 2))

    def forward(self, x):
        b, c, h, w = x.shape # B C H W
        
        if self.F_seq != x.device:
            self.F_seq = self.F_seq.to(x.device)

        if self.training:
            if self.iteration_num != 0:
                F_detach = self.F_seq.detach()
                F_bar = F_detach.mean(0) * self.N / max(1, self.G_num) # G C
                F_nG = F_bar.repeat(b, 1, 1) # B G C
            else:
                self.iteration_num += 1
                h_idx = torch.randint(0, h, (self.G,), requires_grad=False)
                w_idx = torch.randint(0, w, (self.G,), requires_grad=False)
                F_nG = torch.zeros(b, self.G, c).to(x.device)
                for i in range(self.G):
                    F_nG[:, i, :] = x[:, :, h_idx[i], w_idx[i]]
            self.save = False
            for i in range(self.iter):
                # print(F_nG.max(1).values)
                if self.r_func == "dot":
                    P_hat = torch.sum(x.unsqueeze(1) * F_nG.unsqueeze(-1).unsqueeze(-1), 2) / math.sqrt(c) # should change to l1 of F_nG
                    # B G H W
                elif self.r_func == "cosine":
                    P_hat = torch.sum(x.unsqueeze(1) * F_nG.unsqueeze(-1).unsqueeze(-1), 2) / ((F_nG ** 2).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-7)
                elif self.r_func == "pearson":
                    cov = (x - x.mean(1).unsqueeze(1)).unsqueeze(1) * (F_nG - F_nG.mean(-1).unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
                    # B G C H W
                    P_hat = torch.sum(cov / F_nG.std(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 2)
                elif self.r_func == "euclid":
                    P_hat = 1 / (torch.sum((x.unsqueeze(1) - F_nG.unsqueeze(-1).unsqueeze(-1)) ** 2, 2) + 1e-7)
                
                phi = (P_hat == P_hat.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
                
                F_nG = (x.unsqueeze(1) * phi.unsqueeze(2)).sum(-1).sum(-1) / (phi.unsqueeze(2).sum(-1).sum(-1) + 1e-7)
                

            for j in range(b):
                self.F_seq[self.idx, :, :] = F_nG[j, :, :].detach()
                
                self.idx += 1
                if self.idx >= self.N:
                    self.idx = 0

                if self.G_num < self.N:
                    self.G_num += 1
                    
        else:
            if self.version == "train" and not self.save:
                if not os.path.exists(self.save_path + '_' + str(self.save_id) + '.pth'):
                    
                    seq = self.F_seq
                    seq = seq.to("cpu")
                    torch.save(seq, self.save_path + '_' + str(self.save_id) + '.pth')
                    self.save = True
                self.save_id += 1
            elif self.version == "test" and not self.load:
                seq = torch.load(self.save_path + '.pth')
                self.F_seq = seq.to(self.F_seq.device)
                self.load = True

            F_detach = self.F_seq.detach()
            if self.G_num != 0:
                F_bar = F_detach.mean(0) * self.N / max(1, self.G_num) # G C
            else:
                F_bar = F_detach.mean(0)
            F_nG = F_bar.repeat(b, 1, 1) # B G C
            
            if self.r_func == "dot":
                P_hat = torch.sum(x.unsqueeze(1) * F_nG.unsqueeze(-1).unsqueeze(-1), 2) / math.sqrt(c)
                # B G H W
            elif self.r_func == "cosine":
                P_hat = torch.sum(x.unsqueeze(1) * F_nG.unsqueeze(-1).unsqueeze(-1), 2) / (F_nG ** 2).sum(-1).unsqueeze(-1).unsqueeze(-1)
            elif self.r_func == "pearson":
                cov = (x - x.mean(1).unsqueeze(1)).unsqueeze(1) * (F_nG - F_nG.mean(-1).unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
                # B G C H W
                P_hat = torch.sum(cov / F_nG.std(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 2)
            elif self.r_func == "euclid":
                P_hat = 1 / (torch.sum((x.unsqueeze(1) - F_nG.unsqueeze(-1).unsqueeze(-1)) ** 2, 2) + 1e-7)        
            
            phi = (P_hat == P_hat.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
               
        phi_soft = self.avgpool(phi) # B G H W
        F_detach = self.F_seq.detach()
        if self.G_num != 0:
            F_bar = F_detach.mean(0) * self.N / max(1, self.G_num) # G C
        else:
            F_bar = F_detach.mean(0)
        F_bar = F_bar.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # 1 G C 1 1

        x_bar = (phi_soft.unsqueeze(2) * F_bar).sum(1)
        

        return x - self.alpha * x_bar

@MODELS.register_module()
class LCEDHead(BaseDecodeHead): # latent component decomposition simple
    
    def __init__(self,
                 mem_length, # N: the length of memory for save how many samples' components vector
                 G, # G: number of components
                 smooth_kernel, # s: kernel size of avgpool for smoothness
                 iter_train, # iterations for training
                 r_func, # relevancy function
                 save_path,
                 version,
                 alpha,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.icg = ICG(mem_length, G, smooth_kernel, iter_train, r_func, self.channels, save_path, version, alpha)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.icg(out)

        out = self.cls_seg(out)

        return out