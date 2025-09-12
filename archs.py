import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

import timm
from timm.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from pdb import set_trace as st

from kan import KANLinear
from torch.nn import init

from einops import rearrange
import torch.utils.checkpoint as checkpoint


class KANLayer(nn.Module):
    def __init__(self, 
                in_features,
                hidden_features=None,
                out_features=None,
                act_layer=nn.GELU,
                drop=0.,
                no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline=1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
            self,
            img_size=224,
            patch_size=7,
            stride=4,
            in_chans=3,
            embed_dim=768,
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W  

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
    
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        


class MY_Unet(nn.Module):
    def __init__(
            self,
            num_classes,
            in_chans=3,
            deep_supervision=False,
            img_size=224,
            patch_size=16,
            embed_dims=[192, 384, 768],
            no_kan=False,
            drop_rate=0.1,
            drop_patch_rate=0.1,
            norm_layer=nn.LayerNorm,
            depths=[1, 1, 1],
        ):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim//4)
        self.encoder1_x1 = nn.Conv2d(3, kan_input_dim//4, kernel_size=1, bias=True)
        self.encoder1_eca = eca_layer(channel=kan_input_dim//4)
        self.encoder1_reseca = eca_layer(channel=kan_input_dim//4)

        self.encoder2 = ConvLayer(kan_input_dim//4, kan_input_dim//2)
        self.encoder2_x1 = nn.Conv2d(kan_input_dim//4, kan_input_dim//2, kernel_size=1, bias=True)
        self.encoder2_eca = eca_layer(channel=kan_input_dim//2)
        self.encoder2_reseca = eca_layer(channel=kan_input_dim//2)
        
        self.encoder3 = ConvLayer(kan_input_dim//2, kan_input_dim)
        self.encoder3_x1 = nn.Conv2d(kan_input_dim//2, kan_input_dim, kernel_size=1, bias=True)
        self.encoder3_eca = eca_layer(channel=kan_input_dim//2)
        self.encoder3_reseca = eca_layer(channel=kan_input_dim//2)
        
        self.norm2 = norm_layer(embed_dims[0])

        self.norm3 = norm_layer(embed_dims[1])
        self.tobottleneck_eca = eca_layer(channel=embed_dims[1])

        self.norm4 = norm_layer(embed_dims[2])
        self.bottleneck_eca = eca_layer(channel=embed_dims[2])
        self.norm4_b = norm_layer(embed_dims[2])
       
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_patch_rate, sum(depths))]

        self.block0 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[0],
                    drop=drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer
                )
            ]
        )
        self.block1 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[1],
                    drop=drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer
                )
            ]
        )
        self.block2 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[2],
                    drop=drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer
                )
            ]
        )
        self.block2_b = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[2],
                    drop=drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer
                )
            ]
        )
        self.dblock1 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[1],
                    drop=drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer
                )
            ]
        )
        self.dblock0 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[0],
                    drop=drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer
                )
            ]
        )
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]) 
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder1_concat_linear = nn.Linear(embed_dims[1]+embed_dims[1], embed_dims[1])

        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder2_eca = eca_layer(channel=embed_dims[1])
        self.decoder2_x1 = nn.Conv2d(embed_dims[1], embed_dims[0], kernel_size=1, bias=True)
        self.decoder2_concat_linear = nn.Linear(embed_dims[0]+embed_dims[0], embed_dims[0])

        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//2)
        self.decoder3_eca = eca_layer(channel=embed_dims[0])
        self.decoder3_x1 = nn.Conv2d(embed_dims[0], embed_dims[0]//2, kernel_size=1, bias=True)
        self.decoder3_concat_linear = nn.Linear(embed_dims[0]//2+embed_dims[0]//2, embed_dims[0]//2)

        self.decoder4 = D_ConvLayer(embed_dims[0]//2, embed_dims[0]//4)
        self.decoder4_eca = eca_layer(channel=embed_dims[0]//2)
        self.decoder4_x1 = nn.Conv2d(embed_dims[0]//2, embed_dims[0]//4, kernel_size=1, bias=True)
        self.decoder4_concat_linear = nn.Linear(embed_dims[0]//4+embed_dims[0]//4, embed_dims[0]//4)

        self.decoder5 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//4)
        self.decoder5_eca = eca_layer(channel=embed_dims[0]//4)

        self.final = nn.Conv2d(embed_dims[0]//4, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        
    def dual_branch(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        out_x1 = self.encoder1_eca(F.relu(F.max_pool2d(self.encoder1_x1(x), 2, 2)))
        x = torch.add(out, out_x1)

        t1 = self.encoder1_reseca(x)

        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(x), 2, 2))
        out_x1 = self.encoder2_eca(F.relu(F.max_pool2d(self.encoder2_x1(x), 2, 2)))
        x = torch.add(out, out_x1)
        t2 = self.encoder2_reseca(x)


        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(x), 2, 2))
        out_x1 = self.encoder3_eca(F.relu(F.max_pool2d(self.encoder3_x1(x), 2, 2)))
        x = torch.add(out, out_x1)
        t3 = self.encoder3_reseca(x)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.block0):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = self.tobottleneck_eca(x)

        ### Bottleneck
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm4(x)
        skip0 = self.bottleneck_eca(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        for i, blk in enumerate(self.block2_b):
            x = blk(x, H, W)
        x = self.norm4_b(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = torch.add(x, skip0)

        ### Stage 4
        x = F.relu(F.interpolate(self.decoder1(x), scale_factor=(2,2), mode ='bilinear'))
        
        x = torch.add(x, t4)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            x = blk(x, H, W)
        x = self.dnorm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder2(x),scale_factor=(2,2),mode ='bilinear'))
        out_x1 = F.relu(F.interpolate(self.decoder2_x1(self.decoder2_eca(x)),scale_factor=(2,2),mode ='bilinear'))
        x = torch.add(out, out_x1)
        x = torch.add(x, t3)
        
        _,_,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock0):
            x = blk(x, H, W)
        x = self.dnorm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(x),scale_factor=(2,2),mode ='bilinear'))
        out_x1 = F.relu(F.interpolate(self.decoder3_x1(self.decoder3_eca(x)),scale_factor=(2,2),mode ='bilinear'))
        x = torch.add(out, out_x1)
        x = torch.add(x, t2)

        out = F.relu(F.interpolate(self.decoder4(x),scale_factor=(2,2),mode ='bilinear'))
        out_x1 = F.relu(F.interpolate(self.decoder4_x1(self.decoder4_eca(x)),scale_factor=(2,2),mode ='bilinear'))
        x = torch.add(out, out_x1)
        x = torch.add(x, t1)
        
        x = F.relu(F.interpolate(self.decoder5(self.decoder5_eca(x)),scale_factor=(2,2),mode ='bilinear'))

        return self.final(x)

    def forward(self, x):
        x = self.dual_branch(x)
        return x
