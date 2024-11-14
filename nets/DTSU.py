# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .EncoderDecoder import EncoderDecoder
from timm.models.layers import DropPath, to_2tuple
from thop import profile




def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )


class SpectralAttention(nn.Module):
    def __init__(self, in_planes, pool_outsize=4, ratio=16):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 1, 1)
        # self.fc1 =nn.Linear(in_planes, in_planes),
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes , in_planes// 1, 1)
        # self.fc2 = nn.Linear(in_planes, in_planes),
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        c= self.sigmoid(out)
        # attention_weights = c / c.sum(dim=(2, 3), keepdim=True)
        x = x * self.sigmoid(out)
        return x


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )


def repeat_block(block_channel, r, n):
    layers = [
        nn.Sequential(
            SpectralAttention(block_channel, r),
            conv3x3_gn_relu(block_channel, block_channel, r)
        )
        for _ in range(n)]
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self,  in_channel=32, inner_dim = 12 ,r=1,num_blocks=(1, 1, 1, 1, 1, 1)):
        super(Encoder, self).__init__()

        block1_channels = 96
        block2_channels = 128
        block3_channels = 192
        block4_channels = 256
        block5_channels = 320
        block6_channels = 384
        self.encoder_feature = nn.ModuleList([
            conv3x3_gn_relu(in_channel, block1_channels, r),

            repeat_block(block1_channels, r, num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, num_blocks[3]),
            nn.Identity(),
            downsample2x(block4_channels, block5_channels),

            repeat_block(block5_channels, r, num_blocks[4]),
            nn.Identity(),
            downsample2x(block5_channels, block6_channels),

            repeat_block(block6_channels, r, num_blocks[5]),
        ])

        self.reduce_1x1convs = nn.ModuleList([
           nn.Conv2d(block1_channels, inner_dim, 1),
           nn.Conv2d(block2_channels, inner_dim, 1),
           nn.Conv2d(block3_channels, inner_dim, 1),
           nn.Conv2d(block4_channels, inner_dim, 1),
           nn.Conv2d(block5_channels, inner_dim, 1),
           nn.Conv2d(block6_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
           nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
           nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
           nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
           nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
           nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
           nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])

    def forward(self, x, mask=None):

        feat_list = []
        for encoder in self.encoder_feature:
          x = encoder(x)
          if isinstance(encoder, nn.Identity):
            feat_list.append(x)
        feat_list.append(x)
   

        return feat_list

class Decoder(nn.Module):
    def __init__(self, inner_dim=128 ):
        super(Decoder, self).__init__()

        self.tou_1x1convs = nn.ModuleList([
            # nn.Conv2d(64, inner_dim, 1),
            nn.Conv2d(96, inner_dim, 1),
            nn.Conv2d(128, inner_dim, 1),
            nn.Conv2d(192, inner_dim, 1),
            nn.Conv2d(256, inner_dim, 1),
            nn.Conv2d(320, inner_dim, 1),
            nn.Conv2d(384, inner_dim, 1),
            # nn.Conv2d(2048, inner_dim, 1),

        ])  # 示例初始化
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.top_down = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.model = EncoderDecoder(hidden_dim=128, dim_feedforward=1024,
                                    backbone_num_channels=[128,128,128],
                                    # backbone_num_channels=[192,192,192],
                                    dropout=0.1, activation='relu',
                                    num_feature_levels=3,
                                    # num_feature_levels=3,
                                    nhead=8,
                                    num_encoder_layers=1, num_decoder_layers=1,
                                    num_encoder_points=6, num_decoder_points=6,
                                    nclass=24)


    def forward(self, feat_list):
        # 处理输入的特征
        inner_feat_list = [self.tou_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        # 将第一个特征展平并传递给模型
        c6 = inner_feat_list[0].flatten(2)
        x_trans, memory = self.model(inner_feat_list[:3], c6)  # 使用前3个特征
        x_trans = x_trans.permute(0, 2, 1)

        # 索引和形状调整
        x0_index = inner_feat_list[0].shape[-1] * inner_feat_list[0].shape[-2]
        x1_index = inner_feat_list[1].shape[-1] * inner_feat_list[1].shape[-2]

        # 使用 memory 将特征转换成目标形状
        x0 = memory[:, 0:x0_index].permute(0, 2, 1).reshape(
            [inner_feat_list[0].shape[0], 128, inner_feat_list[0].shape[-2], inner_feat_list[0].shape[-1]]
        )
        x1 = memory[:, x0_index:x0_index + x1_index].permute(0, 2, 1).reshape(
            [inner_feat_list[1].shape[0], 128, inner_feat_list[1].shape[-2], inner_feat_list[1].shape[-1]]
        )
        x2 = memory[:, x0_index + x1_index:].permute(0, 2, 1).reshape(
            [inner_feat_list[2].shape[0], 128, inner_feat_list[2].shape[-2], inner_feat_list[2].shape[-1]]
        )

        # 将 x_trans 调整形状
        x_trans = x_trans.reshape(
            [x0.shape[0], 128, x0.shape[-2], x0.shape[-1]]
        )
        x0 = x_trans + x0


        inner_feat_list = inner_feat_list[-3:]
        inner_feat_list.insert(0, x2)
        inner_feat_list.insert(0, x1)
        inner_feat_list.insert(0, x0)

        # 使用 3x3 卷积对特征进行融合
        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i]) + inner_feat_list[i + 1]
            out = self.fuse_3x3convs[i](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]
        return final_feat


class DTSU(nn.Module):

    def __init__(self, encoder,decoder,inner_dim = int(128),

                 **kwargs):
        super().__init__()



        self.cls_pred_conv = nn.Conv2d(inner_dim, 24, 1)


        self.encoder = encoder()
        self.decoder = decoder()


        # 定义反卷积（转置卷积）层
        self.relu = nn.ReLU()

        # self.reset_parameters()

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')

        return lateral + top2x


    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):

        Trans_features = []
        feat_list=self.encoder(x,mask=None)



        final_feat = self.decoder(feat_list)
        logit1 = self.cls_pred_conv(final_feat)






        return  logit1

if __name__=='__main__':
    input=torch.randn(1,32,512,512).to('cuda:0')
    # danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)


    model = DAT(encoder=Encoder,decoder=Decoder).to('cuda:0')
    output = model(input)
    print(model(input).shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    flops, params = profile(model, inputs=(input,))



    params_in_mb = params  / (10**6) # 每个参数通常是4字节

    # 将 FLOP 转换为千亿次每秒（GFLOPS）
    flops_in_gflops = flops / (10**9)

    # 打印估算结果
    print(f"Total parameters: {params_in_mb} MB")
    print(f"Total FLOP: {flops_in_gflops} GFLOPS")