import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmseg.registry import MODELS
from tools.dataset_converters.voc_aug import convert_mat
from .decode_head import BaseDecodeHead
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList
import torch.nn.functional as F
import math
from einops import rearrange, repeat, einsum
from ..utils import resize

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return y

class MultiHeadCCMamba(BaseModule):
    def __init__(self,
                 conv_size, # k
                 dilation, # d
                 num_heads, # n
                 channels): # c
        super().__init__()

        self.k = conv_size
        self.d = dilation
        self.n = num_heads
        self.c = channels

        self.dilated_transform = torch.zeros(conv_size, dtype=torch.int32)
        for i in range(conv_size):
            self.dilated_transform[i] = (i - conv_size // 2) * (dilation + 1)


        self.Proj_h = nn.Conv2d(channels, num_heads * (conv_size + channels), 1)
        self.Proj_w = nn.Conv2d(channels, num_heads * (conv_size + channels), 1)

        A_h = repeat(torch.arange(1, conv_size + 1), 'k -> n c k', n=num_heads, c=channels)
        A_w = repeat(torch.arange(1, conv_size + 1), 'k -> n c k', n=num_heads, c=channels)

        self.A_h_log = nn.Parameter(torch.log(A_h))
        self.A_w_log = nn.Parameter(torch.log(A_w))

        self.SE_h = SEBlock(num_heads, 4)
        self.SE_w = SEBlock(num_heads, 4)

        self.mlp = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x, f):
        if self.A_h_log.device != x.device:
            self.A_h_log = self.A_h_log.to(x.device)
        if self.A_w_log.device != x.device:
            self.A_w_log = self.A_w_log.to(x.device)

        x = F.silu(x)

        A_h = -torch.exp(self.A_h_log.float()) # n c k
        A_w = -torch.exp(self.A_w_log.float()) # n c k

        x_h = x.sum(dim=-1).unsqueeze(-1) # b c h 1
        x_w = x.sum(dim=-2).unsqueeze(-1) # b c w 1

        feat_dbh = self.Proj_h(x_h) # b n(k + c) h 1
        feat_dbw = self.Proj_w(x_w)  # b n(k + c) w 1

        (delta_h, B_h) = feat_dbh.squeeze(-1).split(split_size=[self.n * self.k, self.n * self.c], dim=1) # (b, nk, h), (b, nc, h)
        (delta_w, B_w) = feat_dbw.squeeze(-1).split(split_size=[self.n * self.k, self.n * self.c], dim=1) # (b, nk, w), (b, nc, w)

        delta_h = rearrange(delta_h, 'b (n k) h -> b n k h', n=self.n, k=self.k)
        delta_w = rearrange(delta_w, 'b (n k) w -> b n k w', n=self.n, k=self.k)

        delta_h = F.softplus(delta_h)
        delta_w = F.softplus(delta_w)

        delta_A_h = delta_h.unsqueeze(2) * A_h.unsqueeze(0).unsqueeze(-1) # b n c k h
        delta_A_w = delta_w.unsqueeze(2) * A_w.unsqueeze(0).unsqueeze(-1) # b n c k w

        B_h = rearrange(B_h, 'b (n c) h -> b n c h', n=self.n, c=self.c)
        B_w = rearrange(B_w, 'b (n c) w -> b n c w', n=self.n, c=self.c)

        delta_B_h = delta_h.unsqueeze(2) * B_h.unsqueeze(3) # b n c k h
        delta_B_w = delta_w.unsqueeze(2) * B_w.unsqueeze(3) # b n c k w

        x_h = x_h.squeeze(-1) # b c h
        x_w = x_w.squeeze(-1) # b c w

        attn_h = torch.zeros_like(delta_A_h).to(x.device) # b n c k h
        state_h = torch.zeros(delta_A_h.shape[:-1]).to(x.device)
        for i in range(delta_A_h.shape[-1]):
            state_h = delta_A_h[:, :, :, :, i] * state_h + delta_B_h[:, :, :, :, i] * x_h[:, :, i].unsqueeze(1).unsqueeze(3)
            
            attn_h[:, :, :, :, i] = state_h

        attn_w = torch.zeros_like(delta_A_w).to(x.device)
        state_w = torch.zeros(delta_A_w.shape[:-1]).to(x.device)
        for i in range(delta_A_w.shape[-1]):
            state_w = delta_A_w[:, :, :, :, i] * state_w + delta_B_w[:, :, :, :, i] * x_w[:, :, i].unsqueeze(
                1).unsqueeze(3)
            
            attn_w[:, :, :, :, i] = state_w

        f_h = attn_h[:, :, :, :, -1].unsqueeze(-1).unsqueeze(-1) * f.unsqueeze(1).unsqueeze(3) # b n c k h w
        f_w = attn_w[:, :, :, :, -1].unsqueeze(-1).unsqueeze(-1) * f.unsqueeze(1).unsqueeze(3)

        feat_h = torch.zeros(f_h.shape[0], self.n, self.c, f_h.shape[-2], f_h.shape[-1]).to(x.device) # b, n, c, h, w
        for i in range(self.k):
            mid = torch.zeros_like(feat_h)
            if self.dilated_transform[i] >= 0:
                mid[:, :, :, :f_h.shape[-2] - self.dilated_transform[i], :] = f_h[:, :, :, i, self.dilated_transform[i]:, :]
            else:
                mid[:, :, :, -self.dilated_transform[i]:, :] = f_h[:, :, :, i, :f_h.shape[-2] + self.dilated_transform[i], :]
            feat_h += mid

        feat_w = torch.zeros(f_w.shape[0], self.n, self.c, f_w.shape[-2], f_w.shape[-1]).to(x.device) # b, n, c, h, w
        for i in range(self.k):
            mid = torch.zeros_like(feat_w)
            if self.dilated_transform[i] >= 0:
                mid[:, :, :, :, :f_w.shape[-1] - self.dilated_transform[i]] = f_w[:, :, :, i, :,
                                                                              self.dilated_transform[i]:]
            else:
                mid[:, :, :, :, -self.dilated_transform[i]:] = f_w[:, :, :, i, :,
                                                               :f_w.shape[-1] + self.dilated_transform[i]]
            feat_w += mid

        #### Squeeze and Excitation
        squeeze_h = rearrange(feat_h, 'b n c h w -> b n (c h w)')
        squeeze_h = squeeze_h.mean(-1)

        squeeze_w = rearrange(feat_w, 'b n c h w -> b n (c h w)')
        squeeze_w = squeeze_w.mean(-1)

        squeeze_h = self.SE_h(squeeze_h)
        squeeze_w = self.SE_h(squeeze_w)

        fuse_feat = torch.cat([(feat_h * squeeze_h.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(1), (feat_w * squeeze_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(1)], dim=1) # b 2c h w

        return self.mlp(fuse_feat) + f


@MODELS.register_module()
class CCMambaHead(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', relation=[[1, 2, 3]], kernel_size=[[3, 3, 3]], dilation=[[1, 1, 1]], num_heads=16, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        assert len(relation) == len(kernel_size)
        assert len(relation) == len(dilation)
        for idx in range(len(relation)):
            assert len(relation[idx]) == len(kernel_size[idx])
            assert len(relation[idx]) == len(dilation[idx])
            for i in relation[idx]:
                assert i > idx
            for i in kernel_size[idx]:
                assert i > 0
            for i in dilation[idx]:
                assert i >= 0

        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)
        assert len(relation) < num_inputs

        self.relation = relation
        self.kernel_size = kernel_size
        self.dilation = dilation

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

        self.CCMamba = nn.ModuleList()
        for idx in range(len(relation)):
            for i in range(len(relation[idx])):
                self.CCMamba.append(
                    MultiHeadCCMamba(
                        conv_size=kernel_size[idx][i],
                        dilation=dilation[idx][i],
                        num_heads=num_heads,
                        channels=self.channels
                    )
                )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)


    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        x = []
        for idx in range(len(inputs)):
            x.append(self.convs[idx](inputs[idx]))

        num = 0
        for idx in range(len(self.relation)):
            for i in range(len(self.relation[idx])):
                x[idx] = self.CCMamba[num](x[self.relation[idx][i]], x[idx])

                num += 1

        for idx in range(len(x)):
            x[idx] = resize(
                input=x[idx],
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )

        out = self.fusion_conv(torch.cat(x, dim=1))


        return out

    def forward(self, inputs):
        """Forward function."""
        outputs = self._forward_feature(inputs)

        out = self.cls_seg(outputs)

        return out