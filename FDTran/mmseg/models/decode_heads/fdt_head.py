import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList
import torch.nn.functional as F
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from typing import Tuple, Union
from torch import Tensor
from mmseg.utils import OptConfigType, SampleList

class FDTrans(BaseModule):
    def __init__(self,
                 x1_dims,
                 x2_dims,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 batch_first=True,
                 qkv_bias=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 ):
        super().__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)

        self.feat1_mlp = nn.Conv2d(in_channels=x1_dims, out_channels=embed_dims, kernel_size=1)
        self.feat_i_mlp = nn.Conv2d(in_channels=x2_dims, out_channels=embed_dims, kernel_size=1)

        self.mlp = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x1, x2):
        bs, c, h, w = x2.shape
        h1, w1 = x1.shape[1:3]
        multiple = int(h1 / h)
        if multiple == 0:
            multiple = 1
        x1 = F.interpolate(x1, size=[h * multiple, w * multiple], mode='bilinear')
        x1 = self.feat1_mlp(x1)
        x2 = self.feat_i_mlp(x2)
        bs, c, h, w = x2.shape

        x1_ = x1.permute(0, 2, 3, 1).reshape(bs, h * w, c, -1)
        x2_ = x2.permute(0, 2, 3, 1).reshape(bs, h * w, c)
        a = torch.zeros_like(x2).to(device=x2.device)
        def _inner_forward(x1, x2):
            a = self.attn(self.norm1(x1), self.norm1(x2), self.norm1(x2), identity=x2)
            a = self.ffn(self.norm2(a))
            return a

        for i in range(multiple):
            a_ = _inner_forward(x1_[:, :, :, i], x2_)
            a += a_.reshape(bs, h, w, c).permute(0, 3, 1, 2)

        feat_p = self.mlp(a * x2 / multiple)
        feat_n = x2 - feat_p

        return feat_p, feat_n


@MODELS.register_module()
class FDTHead(BaseDecodeHead):
    def __init__(self, num_heads=1, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.num_feats = len(self.in_index)
        self.FDT = nn.ModuleList()
        self.mlp_p = nn.ModuleList()
        self.mlp_n = nn.ModuleList()
        for i in range(self.num_feats - 1):
            self.FDT.append(FDTrans(self.in_channels[0], self.in_channels[i+1], self.channels, num_heads, self.channels*2))
            self.mlp_p.append(nn.Conv2d(self.channels, self.channels, 3, 1, 1))
            self.mlp_n.append(nn.Conv2d(self.channels, self.channels, 3, 1, 1))

        self.mlp0 = nn.Conv2d(self.in_channels[0], self.channels, 3, 1, 1)

        self.fpn_bottleneck_p = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fpn_bottleneck_n = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_seg_p = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.conv_seg_n = nn.Conv2d(self.channels, 1, kernel_size=1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feat_ps = []
        feat_ns = []
        for i in range(self.num_feats - 1):
            feat_p, feat_n = self.FDT[i](x[0], x[i+1])

            feat_p = self.mlp_p[i](feat_p)
            feat_n = self.mlp_n[i](feat_n)

            feat_p = F.interpolate(feat_p, x[0].shape[2:], mode='bilinear')
            feat_n = F.interpolate(feat_n, x[0].shape[2:], mode='bilinear')

            feat_ps.append(feat_p)
            feat_ns.append(feat_n)

        feat0 = self.mlp0(x[0])
        feat_ps.append(feat0)
        feat_ns.append(feat0)

        pos_outputs = torch.cat(feat_ps, dim=1)
        neg_outputs = torch.cat(feat_ns, dim=1)

        pos_outputs = self.fpn_bottleneck_p(pos_outputs)
        neg_outputs = self.fpn_bottleneck_n(neg_outputs)

        return pos_outputs, neg_outputs

    def forward(self, inputs):
        """Forward function."""
        pos_outputs, neg_outputs = self._forward_feature(inputs)
        if self.dropout is not None:
            pos_outputs = self.dropout(pos_outputs)
            neg_outputs = self.dropout(neg_outputs)
        output_p = self.conv_seg_p(pos_outputs)
        output_n = self.conv_seg_n(neg_outputs)

        output = torch.cat([output_n, output_p], dim=1)
        return output
