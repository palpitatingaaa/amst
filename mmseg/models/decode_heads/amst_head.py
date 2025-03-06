# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmengine.registry import MODELS
from ..utils import resize


class RCU(nn.Module):
    """Residual Convolutional Unit"""

    def __init__(self, channels):
        super(RCU, self).__init__()
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        identity = x

        # First block
        x = self.relu1(x)
        x = self.conv1(x)
        '''x = self.relu2(x)
        x = self.conv2(x)'''
        # Residual connection
        return x + identity
class UnifiedAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(UnifiedAttention, self).__init__()
        # 通道注意力部分
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

        # 空间注意力部分
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_pool = self.global_pool(x)
        x_channel = self.fc1(avg_pool)
        x_channel = self.fc2(x_channel)
        channel_attention = self.sigmoid(x_channel)

        # 空间注意力
        avg_pool_spatial = torch.mean(x, dim=1, keepdim=True)
        max_pool_spatial, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)
        spatial_attention = self.conv(combined)
        spatial_attention = self.sigmoid(spatial_attention)

        # 综合注意力结果
        attention = channel_attention * spatial_attention
        return x * attention


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_concat))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

@MODELS.register_module()
class AMSTHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        # CBAM modules
        self.att = nn.ModuleList()
        for i in range(num_inputs):
            self.att.append(
                CBAM(
                    in_channels=self.in_channels[i]))

        # Initial 1x1 convolutions
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

        # Pre-fusion and post-fusion RCU modules
        self.pre_rcus = nn.ModuleList()
        self.post_rcus = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.pre_rcus.append(RCU(channels=self.channels))
            self.post_rcus.append(RCU(channels=self.channels))

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)

        # Apply attention and initial convolutions
        feats = []
        for i in range(len(inputs)):
            x = self.att[i](inputs[i])
            x = self.convs[i](x)
            feats.append(x)

        # Progressive fusion with RCU
        for i in range(len(feats) - 1, 0, -1):
            curr_feat = feats[i]

            # 1. Apply pre-fusion RCU
            curr_feat = self.pre_rcus[i - 1](curr_feat)

            # 2. Upsample to match spatial dimensions
            curr_feat = resize(
                curr_feat,
                size=feats[i - 1].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )

            # 3. Add with previous stage features
            curr_feat = curr_feat + feats[i - 1]

            # 4. Apply post-fusion RCU
            curr_feat = self.post_rcus[i - 1](curr_feat)

            # Update features
            feats[i - 1] = curr_feat

        # Final classification
        out = self.cls_seg(feats[0])
        return out
'''@MODELS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)
        self.att = nn.ModuleList()
        for i in range(num_inputs):
            self.att.append(
                UnifiedAttention(
                    in_channels=self.in_channels[i]))
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
        self.fusion_convs = nn.ModuleList()
        for i in range(num_inputs - 1):  # For fusing outputs from higher to lower layers
            self.fusion_convs.append(
                ConvModule(
                    in_channels=self.channels * 2,
                    out_channels=self.channels,
                    kernel_size=3,
                    norm_cfg=self.norm_cfg))

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        x4 = self.att[3](inputs[3])
        x3 = self.att[2](inputs[2])
        x2 = self.att[1](inputs[1])
        x1 = self.att[0](inputs[0]) 

        x4 = self.convs[3](x4)
        x4  = resize(x4, size=inputs[2].shape[2:], mode=self.interpolate_mode, align_corners=self.align_corners)

        x3 = self.convs[2](x3)
        x3 = torch.cat([x3, x4], dim=1)
        x3 = self.fusion_convs[0](x3)
        x3 = resize(x3, size=inputs[1].shape[2:], mode=self.interpolate_mode, align_corners=self.align_corners)

        x2 = self.convs[1](x2)
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.fusion_convs[1](x2)
        x2 = resize(x2, size=inputs[0].shape[2:], mode=self.interpolate_mode, align_corners=self.align_corners)

        x1 = self.convs[0](x1)
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.fusion_convs[2](x1)
        out = self.cls_seg(x1)
        return out
'''