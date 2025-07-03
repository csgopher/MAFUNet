import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class HAM(nn.Module):
    def __init__(self, in_channels, scale=4, d_state=16, d_conv=4, stride=1, expand=2):
        super(HAM, self).__init__()
        self.channels = in_channels
        self.scale = scale
        self.stride = stride
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=in_channels // scale,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(self.scale)
        ])
        self.cdgf_blocks = nn.ModuleList([
            CDGF() for _ in range(self.scale)
        ])

    def forward(self, x):
        spx = torch.split(x, self.channels // self.scale, dim=1)
        features = []
        prev_feat = None
        for i, (block, cdgf) in enumerate(zip(self.mamba_blocks, self.cdgf_blocks)):
            x_i = spx[i]
            x_i = x_i + prev_feat if prev_feat is not None else x_i

            b, c, h, w = x_i.shape
            sp = x_i.view(b, c, -1).permute(0, 2, 1)
            x_m = block(sp).permute(0, 2, 1).view(b, c, h, w)

            fused = cdgf([x_i, x_m])
            features.append(fused)
            prev_feat = fused

        return torch.cat(features, dim=1)


class CDGF(nn.Module):
    def __init__(self):
        super(CDGF, self).__init__()

    def channel_attention(self, feat):
        return torch.sigmoid(F.adaptive_avg_pool2d(feat, (1, 1)))

    def forward(self, features):
        if not features or len(features) != 2:
            return None
        x_i, x_m = features
        b, c, h, w = x_i.shape

        fused_global = torch.max(
            x_i.view(b, -1, h * w),
            x_m.view(b, -1, h * w)
        ).view(b, -1, h, w)

        w_local = self.channel_attention(x_i)
        w_global = self.channel_attention(fused_global)

        fused_feat = (w_local * x_i + w_global * fused_global)
        return fused_feat


class SDGF(nn.Module):
    def __init__(self):
        super(SDGF, self).__init__()
        self.conv_compress = nn.Conv2d(1, 1, kernel_size=7, padding=3)

    def spatial_attention(self, feat):
        feat_compressed = feat.mean(dim=1, keepdim=True)  # B*1*H*W
        feat_spatial = self.conv_compress(feat_compressed)
        return torch.sigmoid(feat_spatial)

    def forward(self, features):
        if not features or len(features) != 2:
            return None

        x_i, x_m = features

        x_i_spatial = x_i.mean(dim=1, keepdim=True)  # B*1*H*W
        x_m_spatial = x_m.mean(dim=1, keepdim=True)  # B*1*H*W

        fused_global = torch.max(x_i_spatial, x_m_spatial)

        w_local = self.spatial_attention(x_i)  # B*1*H*W
        w_global = self.spatial_attention(fused_global)  # B*1*H*W

        w_local = w_local.expand_as(x_i)
        w_global = w_global.expand_as(x_i)

        fused_feat = (w_local * x_i + w_global * x_m)

        return fused_feat

