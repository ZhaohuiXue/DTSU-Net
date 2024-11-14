import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.container import ModuleList
import copy
import torch
import torch.nn.functional as F
# from torch.nn.init import xavier_uniform_, normal_, constant_
# from torch.nn.parameter import Parameter
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    bs, Len_v, n_head, c = value.shape
    _, Len_q, n_head, n_levels, n_points, _ = sampling_locations.shape
    value_list = value.split(value_spatial_shapes.prod(1).tolist(), dim=1)

    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes.tolist()):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * n_head, c, h, w)


        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)

        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(bs * n_head, 1, Len_q, n_levels * n_points)


    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, lr_mult=0.1):
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        #
        # self.attention_weights_list = []
        # self.sampling_points_list = []

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        print(grid_init.shape)
        max_values = grid_init.abs().max(-1, keepdim=True).values
        # print(max_values)  # 输出最大值
        # print(max_values.shape)  # 输出最大值的形状
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).view(1, 1, -1, 1)
        grid_init *= scaling

        self.sampling_offsets.bias.data = grid_init.flatten()
        # attention_weights
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        # assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.to('cuda:0')  # 将 value_mask 移到相同的设备上
            value_mask = value_mask.to(value.dtype).unsqueeze(-1)
            value *= value_mask

        value = value.view(bs, Len_v, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)

        attention_weights = self.attention_weights(query).view(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        # print(attention_weights)

        attention_weights = F.softmax(attention_weights, -1).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)
        # print(attention_weights)
        offset_normalizer = value_spatial_shapes.flip([1]).view(
            1, 1, 1, self.num_levels, 1, 2)
        sampling_offsets = sampling_offsets.to('cuda:0')
        offset_normalizer = offset_normalizer.to('cuda:0')
        reference_points = reference_points.to('cuda:0')

        sampling_locations = reference_points.view(bs, Len_q, 1, self.num_levels, 1, 2) \
                             + sampling_offsets / offset_normalizer
        # image_path = '1.png'
        # image = Image.open(image_path)
        #
        # image_array = np.array(image)
        #     for i in range(1):
        #      for k in range(8):
        # # 获取指定位置的采样点坐标
        #         sample_point_coordinates = (sampling_locations[0, 128, k, i, :, :]).cpu().detach().numpy()
        #         attention_values = attention_weights[0, 128, k, i, :].cpu().detach().numpy()
        #         sample_point_coordinates = (sample_point_coordinates * np.array((16,16))).astype(int)
        #         x_coordinates = sample_point_coordinates[:, 0]
        #         y_coordinates = sample_point_coordinates[:, 1]
        #         point_sizes = attention_values * 1000
        #         plt.imshow(image_array)
        #         plt.scatter(x_coordinates, y_coordinates, c='darkred', s=point_sizes, cmap='viridis', alpha=0.7)
        #         np.savetxt(f'sample_points_attention_group_{i}.txt', np.column_stack((sample_point_coordinates, attention_values)), delimiter=' ')
        # plt.title('Sample Points Visualization')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.legend()
        # plt.show()
        # sampling_locations_np = sampling_locations.cpu().numpy()
        # num_points = sampling_locations_np.shape[4]
        #

        output = deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.1, activation="relu", n_levels=4,
                 n_points=4):
        super(TransformerEncoderLayer, self).__init__()
        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.conv0 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.GELU())

        self.conv1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.GELU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.GELU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.GELU())

        self._reset_parameters()

    def _reset_parameters(self):
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear1.bias.data.zero_()
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.bias.data.zero_()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def seq2_2D(self, src, spatial_shapes):
        bs, hw, c = src.shape
        h = w = int(math.sqrt(hw))

        x0_index = spatial_shapes[0][0].item() * spatial_shapes[0][1].item()  # 32 * 32 = 1024
        x1_index = spatial_shapes[1][0].item() * spatial_shapes[1][1].item()  # 16 * 16 = 256
        x2_index = spatial_shapes[2][0].item() * spatial_shapes[2][1].item()
        # x3_index = spatial_shapes[3][0].item() * spatial_shapes[3][1].item()# 8 * 8 =64

        x0 = src[:, 0:x0_index].transpose(1, 2).reshape(
            [bs, c, spatial_shapes[0][0], spatial_shapes[0][1]])  # shape=[4, 256, 32, 32]
        x1 = src[:, x0_index:x0_index + x1_index].transpose(1, 2).reshape(
            [bs, c, spatial_shapes[1][0], spatial_shapes[1][1]])  # shape=[4, 256, 16, 16]
        x2 = src[:, x0_index + x1_index:x0_index + x1_index+x2_index:].transpose(1, 2).reshape(
            [bs, c, spatial_shapes[2][0], spatial_shapes[2][1]])  # shape=[4, 256, 8, 8]
        # x3 = src[:, x0_index + x1_index+x2_index::].transpose(1, 2).reshape(
        #     [bs, c, spatial_shapes[3][0], spatial_shapes[3][1]])  # shape=[4, 256, 8, 8]

        return x0, x1, x2

    def forward(self, src, reference_points, spatial_shapes, src_mask=None, pos_embed=None):
        x0, x1, x2 = self.seq2_2D(src, spatial_shapes)

        src0 = self.conv0(x0) + x0
        src1 = self.conv1(x1) + x1
        src2 = self.conv2(x2) + x2
        # src3 = self.conv2(x3) + x3

        src0 = src0.flatten(2).transpose(1, 2)  # (bs,c,h,w) ->(bs,h*w,c)
        src1 = src1.flatten(2).transpose(1, 2)  # (bs,c,h,w) ->(bs,h*w,c)
        src2 = src2.flatten(2).transpose(1, 2)
        # src3 = src3.flatten(2).transpose(1, 2)# (bs,c,h,w) ->(bs,h*w,c)

        src_flatten = [src0, src1, src2 ]
        src_flatten = torch.cat(src_flatten, 1)  # [4, 5376, 256]

        src2 = self.self_attn(self.with_pos_embed(src, pos_embed), reference_points, src, spatial_shapes, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        src = src + src_flatten
        return src
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []

        for i, (H, W) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H),
                torch.linspace(0.5, W - 0.5, W))
            # print((valid_ratios[:, :, i, 1] * H))
            a=(valid_ratios[:, :, i, 1] * H)
            c=(valid_ratios[:, :, i, 0] * W)
            # b=ref_y.reshape(1, -1)

            ref_y = ref_y.reshape(1, -1) / a
            ref_x = ref_x.reshape(1, -1) / c
            reference_points.append(torch.stack((ref_x, ref_y), dim=-1))

        reference_points = torch.cat(reference_points, dim=1).unsqueeze(2)
        reference_points = reference_points * valid_ratios
        return reference_points

    def forward(self, src, spatial_shapes, src_mask=None, pos_embed=None, valid_ratios=None):
        output = src
        if valid_ratios is None:
            valid_ratios = torch.ones([src.shape[0], spatial_shapes.shape[0], 2])
        # [bs,num_features,2]

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)
        for layer in self.layers:
            output = layer(output, reference_points, spatial_shapes, src_mask, pos_embed)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.1, activation="relu", n_levels=3,
                 n_points=4):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, 1, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes, memory_mask=None, query_pos_embed=None):
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2, _ = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
                               memory_spatial_shapes, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, H, W):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H),
                                      torch.linspace(0.5, W - 0.5, W))
        ref_y = ref_y.flatten().unsqueeze(0) / H
        ref_x = ref_x.flatten().unsqueeze(0) / W
        ref = torch.stack((ref_x, ref_y), dim=-1)

        reference_points = ref.unsqueeze(2)
        reference_points = reference_points * valid_ratios

        return reference_points

    def forward(self, tgt, memory, reference_points, memory_spatial_shapes, memory_mask=None, query_pos_embed=None,
                valid_ratios=None):

        output = tgt
        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(output, reference_points, memory, memory_spatial_shapes, memory_mask, query_pos_embed)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoder1(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super(TransformerDecoder1, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, H, W):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H),
                                      torch.linspace(0.5, W - 0.5, W))
        ref_y = ref_y.flatten().unsqueeze(0) / H
        ref_x = ref_x.flatten().unsqueeze(0) / W
        ref = torch.stack((ref_x, ref_y), dim=-1)

        reference_points = ref.unsqueeze(2)
        reference_points = reference_points * valid_ratios

        return reference_points

    def forward(self, tgt, memory, reference_points, memory_spatial_shapes, memory_mask=None, query_pos_embed=None,
                valid_ratios=None):

        output = tgt
        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(output, reference_points, memory, memory_spatial_shapes, memory_mask, query_pos_embed)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class PositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None, embed_type='sine',
                 num_embeddings=50, offset=0.):
        super(PositionEmbedding, self).__init__()
        assert embed_type in ['sine', 'learned']

        self.embed_type = embed_type
        self.offset = offset
        self.eps = 1e-6
        if self.embed_type == 'sine':
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            if scale is not None and normalize is False:
                raise ValueError("normalize should be True if scale is passed")
            if scale is None:
                scale = 2 * math.pi
            self.scale = scale
        elif self.embed_type == 'learned':
            self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
            self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        else:
            raise ValueError(f"not supported {self.embed_type}")

    def forward(self, mask):
        """
        Args:
            mask (Tensor): [B, H, W]
        Returns:
            pos (Tensor): [B, C, H, W]
        """
        assert mask.dtype == torch.bool
        if self.embed_type == 'sine':
            mask = mask.to(torch.float32)
            y_embed = mask.cumsum(1, dtype=torch.float32)
            x_embed = mask.cumsum(2, dtype=torch.float32)
            if self.normalize:
                y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
                x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

            dim_t = 2 * (torch.arange(self.num_pos_feats) // 2).to(torch.float32)
            dim_t = self.temperature ** (dim_t / self.num_pos_feats)

            pos_x = x_embed.unsqueeze(-1) / dim_t
            pos_y = y_embed.unsqueeze(-1) / dim_t
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            return pos

        elif self.embed_type == 'learned':
            h, w = mask.shape[-2:]
            i = torch.arange(w)
            j = torch.arange(h)
            x_emb = self.col_embed(i)
            y_emb = self.row_embed(j)
            pos = torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1).permute(2, 0, 1).unsqueeze(0).expand(mask.shape[0], -1, -1, -1)
            return pos
        else:
            raise ValueError(f"not supported {self.embed_type}")


class EncoderDecoder(nn.Module):
    def __init__(self, num_queries=256, position_embed_type='sine', return_intermediate_dec=False,
                 backbone_num_channels=[256, 256, 256, 256], num_feature_levels=4, nclass=6,
                 num_encoder_points=4, num_decoder_points=4, hidden_dim=128, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=4, dim_feedforward=2048, dropout=0.1, activation="relu", lr_mult=0.1

                 ):

        super(EncoderDecoder, self).__init__()
        # backbone_num_channels=[256, 256, 256]
        # nhead=8
        # dim_feedforward=1024
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_num_channels) <= num_feature_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)


        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation,
                                                num_feature_levels, num_encoder_points)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation,
                                                num_feature_levels, num_decoder_points)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self. \
            decoder1 = TransformerDecoder1(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        # self.query_pos_embed = nn.Embedding(, hidden_dim)
        self.query_pos_embed = nn.Embedding(256, hidden_dim)
        self.query_pos_embed0 = nn.Embedding(256, hidden_dim)
        self.query_pos_embed1 = nn.Embedding(1024, hidden_dim)
        self.query_pos_embed2 = nn.Embedding(4096, hidden_dim)

        self.reference_points = nn.Linear(hidden_dim, 2)
        self.input_proj = nn.ModuleList()
        for in_channels in backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim)))

        in_channels = backbone_num_channels[-1]
        for _ in range(num_feature_levels - len(backbone_num_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1,
                              ),
                    nn.GroupNorm(32, hidden_dim)))
            in_channels = hidden_dim

        self.position_embedding = PositionEmbedding(hidden_dim // 2,
                                                    normalize=True if position_embed_type == 'sine' else False,
                                                    embed_type=position_embed_type, offset=-0.5)
        self._reset_parameters()

    def _reset_parameters(self):
        init.normal_(self.level_embed.weight)
        init.normal_(self.tgt_embed.weight)
        init.normal_(self.query_pos_embed.weight)
        init.xavier_uniform_(self.reference_points.weight)
        self.reference_points.bias.data.fill_(0)  # 设置偏置为常量值
        for l in self.input_proj:
            init.xavier_uniform_(l[0].weight)
            l[0].bias.data.fill_(0)  # 设置偏置为常量值

    def _get_valid_ratio(self, mask):
        mask = mask.to(torch.float32)
        _, H, W = mask.shape
        valid_ratio_h = torch.sum(mask[:, :, 0], dim=1) / H
        valid_ratio_w = torch.sum(mask[:, 0, :], dim=1) / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, src_feats, src_psp, src_mask=None):
        srcs = []

        # print(src_feats[0].shape)
        # print(src_feats[1].shape)
        # print(src_feats[2].shape)
        for i in range(len(src_feats)):
            srcs.append(self.input_proj[i](src_feats[i]))

        if self.num_feature_levels > len(srcs):
            len_srcs = len(srcs)
            for i in range(len_srcs, self.num_feature_levels):
                if i == len_srcs:
                    srcs.append(self.input_proj[i](src_feats[-1]))
                else:
                    srcs.append(self.input_proj[i](srcs[-1]))

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for level, src in enumerate(srcs):
            bs, c, h, w = src.shape
            spatial_shapes.append([h, w])
            src = src.flatten(2).transpose(1, 2)  # (bs,c,h,w) ->(bs,h*w,c)
            src_flatten.append(src)
            if src_mask is not None:
                mask = F.interpolate(src_mask.unsqueeze(0).to(src.dtype), size=(h, w))[0].to(torch.bool)
            else:
                mask = torch.ones([bs, h, w], dtype=torch.bool)
            # print(1)
            # print(mask.shape)
            valid_ratio = self._get_valid_ratio(mask)
            valid_ratios.append(valid_ratio)


            # valid_ratios.append(self._get_valid_ratio(mask))
            # print(self._get_valid_ratio(mask).shape)

            # position_embedding = sine + cosine
            pos_embed = self.position_embedding(mask).flatten(2).transpose(1, 2)  # (bs,c,h,w) ->(bs,h*w,c)
            # pos_embed = self.position_embedding(mask).flatten(1).transpose(0, 1)  # (bs,c,h,w) ->(bs,h*w,c)
            # print(pos_embed.shape)

            pos_embed = pos_embed.to(self.level_embed.weight.device)
            lvl_pos_embed = pos_embed + self.level_embed.weight[level].view(1, 1, -1)  # (bs,h*w,c)+(1,1,256)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask = mask.to(src.dtype).view(bs, -1)  # (bs, h, w) -> (bs, h * w)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        masks = mask_flatten# [4, 5376, 256]

        mask_flatten = torch.cat(mask_flatten, 1)
        # print(mask[0].shape)
        # print(mask[1].shape)
        # print(mask[2].shape)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # position + level embed
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.int64)
        valid_ratios = torch.stack(valid_ratios, dim =1)

        # memory = self.encoder(src_flatten, spatial_shapes, mask=mask_flatten, src_key_padding_mask=None,
        #                       pos=lvl_pos_embed_flatten)

        memory1 = self.encoder(src_flatten, spatial_shapes, mask_flatten, lvl_pos_embed_flatten, valid_ratios)
        # output_tensor = self.max_pooling_layer(memory1)
        # mask_flatten = self.max_pooling_layer(mask_flatten)
        # lvl_pos_embed_flatten= self.max_pooling_layer(lvl_pos_embed_flatten)
        memory2 = self.encoder(memory1, spatial_shapes, mask_flatten, lvl_pos_embed_flatten, valid_ratios)
        memory = self.encoder(memory2, spatial_shapes, mask_flatten, lvl_pos_embed_flatten, valid_ratios)
        x0_index = src_feats[0].shape[-1] * src_feats[0].shape[-2]
        x1_index = src_feats[1].shape[-1] * src_feats[1].shape[-2]
        x2_index = src_feats[2].shape[-1] * src_feats[2].shape[-2]
        x0 = memory[:, 0:x0_index]
        x1 = memory[:, x0_index:x0_index + x1_index]
        x2 = memory[:, x0_index + x1_index:x0_index + x1_index + x2_index:]
        # x3 = memory[:, x0_index + x1_index + x2_index::]
        # x3 = memory[:, x0_index + x1_index + x2_index::].permute(0, 2, 1).reshape(
        #     [src_feats[2].shape[0], 256, src_feats[3].shape[-2], src_feats[3].shape[-1]])
        # x_out3 = self.conv(x3)
        # x_out3 = F.interpolate(x_out3, size=src_feats[2].shape[2:], mode='bilinear', align_corners=True)
        # a=x2.shape[2]


        bs, _, c = memory.shape  # [8, 5376, 256]
        query_embed = self.query_pos_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        query_embed0 = self.query_pos_embed0.weight.unsqueeze(0).expand(bs, -1, -1)
        query_embed1 = self.query_pos_embed1.weight.unsqueeze(0).expand(bs, -1, -1)
        query_embed2 = self.query_pos_embed2.weight.unsqueeze(0).expand(bs, -1, -1)

        reference_points = torch.sigmoid(self.reference_points(query_embed))
        reference_points0 = torch.sigmoid(self.reference_points(query_embed0))
        reference_points1 = torch.sigmoid(self.reference_points(query_embed1))
        reference_points2 = torch.sigmoid(self.reference_points(query_embed2))


        # [4, 110, 256]
        reference_points = reference_points.to('cuda:0')  # 将 reference_points 移到 CUDA 设备上
        valid_ratios = valid_ratios.to('cuda:0')  # 将 valid_ratios 移到 CUDA 设备上
        reference_points_input = reference_points.unsqueeze(2) * valid_ratios[:, 0:1, :].unsqueeze(1)
        reference_points_input0 = reference_points0.unsqueeze(2) * valid_ratios[:, 0:1, :].unsqueeze(1)
        reference_points_input1 = reference_points1.unsqueeze(2) * valid_ratios[:, 0:1, :].unsqueeze(1)
        reference_points_input2 = reference_points2.unsqueeze(2) * valid_ratios[:, 0:1, :].unsqueeze(1)
        # reference_points_input1 = reference_points.unsqueeze(2) * valid_ratio.unsqueeze(1)
        sub_tensor = spatial_shapes[2:3, :]
        sub_tensor1 = spatial_shapes[1:2, :]
        sub_tensor2 = spatial_shapes[0:1, :]

        src_psp = src_psp.transpose(1, 2)
        # mask0 = mask_flatten[:, 0:x0_index]
        # hs = self.decoder(src_psp, memory, reference_points_input, spatial_shapes, mask_flatten, query_embed,
        #                   valid_ratios)
        hs0 = self.decoder1(x0, x1,  reference_points_input, sub_tensor1, masks[1], query_embed,
                            valid_ratios
                            )
        hs1 = self.decoder1(hs0, x2,  reference_points_input0, sub_tensor, masks[2], query_embed0,
                            valid_ratios
                            )
        # hs = self.decoder1(hs1, x2,  reference_points_input, sub_tensor, masks[2], query_embed,
        #                    valid_ratios
        #                    )
        # AUX= hs0+ hs1+  hs.squeeze()

        return hs1, memory