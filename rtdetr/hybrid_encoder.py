"""by lyuwenyu
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from .utils import get_activation

__all__ = ["HybridEncoder"]


class ConvNormLayer(nn.Module):
    def __init__(
        self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=None,
        act="silu",
    ):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(
            in_channels, hidden_channels, 1, 1, bias=bias, act=act
        )
        self.conv2 = ConvNormLayer(
            in_channels, hidden_channels, 1, 1, bias=bias, act=act
        )
        self.bottlenecks = nn.Sequential(
            *[
                RepVggBlock(hidden_channels, hidden_channels, act=act)
                for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(
                hidden_channels, out_channels, 1, 1, bias=bias, act=act
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(q, k, pos_embed):
        if pos_embed is not None:
            q = q + pos_embed

            k = k + pos_embed

            return q, k
        else:
            return q, k

    def forward(self, q, k, v, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = v
        if self.normalize_before:
            q = self.norm1(q)
            k = self.norm1(k)
            v = self.norm1(v)
        q, k = self.with_pos_embed(q, k, pos_embed)
        src, attn_weights = self.self_attn(q, k, value=v, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src, attn_weights


class TransformerUserEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1_q = nn.LayerNorm(d_model)
        self.norm1_kv = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(q, pos_embed):
        if pos_embed is not None:
            q = q + pos_embed
            return q
        else:
            return q

    def forward(self, q, k, v, src_mask=None, pos_embed=None) -> torch.Tensor:
        if self.normalize_before:
            q = self.norm1_q(q)
            k = self.norm1_kv(k)
            v = self.norm1_kv(v)
        q = self.with_pos_embed(q, pos_embed)
        src, attn_weights = self.self_attn(q, k, value=v, attn_mask=src_mask)

        src = self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1_q(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, q, k=None, v=None, src_mask=None, pos_embed=None) -> torch.Tensor:
        if k is None:
            k = q
        if v is None:
            v = q

        for layer in self.layers:
            output, attn_weights = layer(
                q, k, v, src_mask=src_mask, pos_embed=pos_embed
            )

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class HybridEncoder(nn.Module):
    __share__ = ["num_classes"]

    def __init__(
        self,
        num_classes=80,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        user_input_pooling="C2",
        inter_dropout=0.0,
        viz_attn=False,
        lump_edition=False,
    ):
        super().__init__()
        self.lump_edition = lump_edition
        self.viz_attn = viz_attn
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.num_classes = num_classes
        self.user_input_pooling = user_input_pooling

        # self.norm_batch = nn.BatchNorm2d(2)

        self.norm_batch = nn.ModuleList()
        for in_channel in in_channels:
            self.norm_batch.append(nn.BatchNorm2d(2))

        # Label embeddings
        self.label_embeddings = [
            nn.Parameter(torch.randn(num_classes, self.hidden_dim))
            for _ in range(len(in_channels))
        ]

        self.attn_conv = nn.ModuleList()
        for in_channel in in_channels:
            self.attn_conv.append(
                nn.Sequential(
                    # nn.Conv2d(2, 2, kernel_size=3, bias=False, padding=1),
                    # get_activation(enc_act),
                    nn.Conv2d(2, 1, kernel_size=3, bias=False, padding=1),
                    get_activation(enc_act),
                    # nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
                )
            )

        # top-down fpn
        self.lateral_convs_attn = nn.ModuleList()
        self.fpn_blocks_attn = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs_attn.append(ConvNormLayer(2, 2, 1, 1, act=act))
            self.fpn_blocks_attn.append(
                CSPRepLayer(
                    2 * 2,
                    2,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                )
            )

        # bottom-up pan
        self.downsample_convs_attn = nn.ModuleList()
        self.pan_blocks_attn = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs_attn.append(ConvNormLayer(2, 2, 3, 2, act=act))
            self.pan_blocks_attn.append(
                CSPRepLayer(
                    2 * 2,
                    2,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                )
            )

        # self.linear_mask = nn.ModuleList()
        # for in_channel in in_channels:
        #     self.linear_mask.append(
        #         nn.Sequential(
        #             # nn.Conv2d(
        #             #     hidden_dim,
        #             #     int(hidden_dim / 2),
        #             #     kernel_size=3,
        #             #     bias=False,
        #             #     padding=1,
        #             # ),
        #             # get_activation(enc_act),
        #             # nn.Conv2d(
        #             #     int(hidden_dim / 2), 1, kernel_size=3, bias=False, padding=1
        #             # ),
        #             nn.Linear(hidden_dim, int(hidden_dim / 2), bias=False),
        #             get_activation(enc_act),
        #             nn.Linear(int(hidden_dim / 2), int(hidden_dim / 2), bias=False),
        #             get_activation(enc_act),
        #             nn.Linear(int(hidden_dim / 2), 1, bias=False),
        #             nn.Sigmoid(),
        #         )
        #     )

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
            )

        self.i_attn_weights_dropout = nn.Dropout(inter_dropout)

        # self.layer_norm = nn.LayerNorm(hidden_dim)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )

        # user_encoder_layer = TransformerUserEncoderLayer(
        #     hidden_dim,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     activation=enc_act,
        # )

        # self.interactive_cross_attention_layer = nn.ModuleList(
        #     [
        #         TransformerEncoder(copy.deepcopy(user_encoder_layer), 1)
        #         for _ in range(len(in_channels))
        #     ]
        # )

        self.interactive_self_attention_layer = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ]
        )

        # self.interactive_self_a_cross_attention_layer = nn.ModuleList(
        #     [
        #         TransformerEncoder(copy.deepcopy(encoder_layer), 1)
        #         for _ in range(len(in_channels))
        #     ]
        # )

        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ]
        )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act)
            )
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                )
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                )
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    def build_2d_sincos_position_embedding(
        self,
        w,
        h,
        embed_dim=256,
        temperature=10000.0,
    ):
        """ """

        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        # print(grid_w.shape)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        pos_emb = torch.concat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None, :, :]

        return pos_emb

    def input_proj_user(self, bbox_feats, indx, labels):

        encoded_batch_feats_C2 = []
        for batch_i, bbox_batch in enumerate(bbox_feats):

            encoded_bbox_feats_C2 = {}
            for box_i, bbox in enumerate(bbox_batch[indx]):
                bbox = bbox.unsqueeze(0)
                self.input_proj.eval()
                encoded_box = self.input_proj[indx](bbox)
                self.input_proj.train()
                label = labels[batch_i][box_i]

                encoded_box = encoded_box.flatten(2).permute(0, 2, 1)

                class_ = int(label.cpu().numpy())

                if class_ in encoded_bbox_feats_C2:
                    encoded_bbox_feats_C2[class_].append(encoded_box)
                else:
                    encoded_bbox_feats_C2[class_] = [encoded_box]

            encoded_bbox_feats_C2_comb = {}
            classes = [k for k in range(self.num_classes)]

            for label, encoded_boxes in encoded_bbox_feats_C2.items():
                encoded_boxes = torch.concat(encoded_boxes, dim=1)

                label_embed = (
                    self.label_embeddings[indx][label, :].unsqueeze(0).unsqueeze(0)
                ).to(encoded_boxes.device)

                encoded_boxes = torch.cat([label_embed, encoded_boxes], dim=1).to(
                    encoded_boxes.device
                )

                encoded_bbox_feats_C2_comb[label] = encoded_boxes
                classes.pop(classes.index(label))

            for label in classes:
                encoded_bbox_feats_C2_comb[label] = (
                    self.label_embeddings[indx][label, :].unsqueeze(0).unsqueeze(0)
                ).to(labels.device)

            encoded_batch_feats_C2.append(encoded_bbox_feats_C2_comb)

        return encoded_batch_feats_C2

    def encode_and_compute_distances(
        self,
        img_feats,
        bbox_feats,
        enc_ind,
        device,
        softmax=False,
        cargs=None,
        sel_cls=None,
    ):
        """Encodes bounding box features and computes distances for attention heatmap."""

        # print(f"sel_cls: {sel_cls}")

        cls_token = (
            self.label_embeddings[enc_ind][sel_cls, :]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        if len(bbox_feats) == 0:
            encoded_boxes = cls_token
        else:

            bbox_feats = torch.cat(bbox_feats, dim=1).to(bbox_feats[0].device)
            encoded_boxes = torch.cat([cls_token, bbox_feats], dim=1)

        if enc_ind in self.use_encoder_idx and not cargs.no_iut:
            idx = self.use_encoder_idx.index(enc_ind)
            self_attended_boxes, _ = self.interactive_self_attention_layer[idx](
                encoded_boxes
            )
        else:
            self_attended_boxes = encoded_boxes

        # encoded_boxes = encoded_boxes[:, :-1]
        self_attended_boxes = self_attended_boxes.squeeze(0).T
        distances = torch.mm(img_feats, self_attended_boxes)
        if cargs.norm_dist:
            distances = (distances - distances.min()) / (
                distances.max() - distances.min()
            )

        if softmax:
            distances = F.softmax(distances, dim=1)[:, :-1].max(1).values
        else:
            distances = distances.max(1).values

        return distances, encoded_boxes

    def interactive_cross_attention_heatmap(
        self,
        img_feats,
        bbox_feats,
        enc_ind,
        pos_embed=None,
        softmax=False,
        cargs=None,
        selected_class=None,
    ):

        attn_weights = torch.zeros(img_feats.shape[0], 2, img_feats.shape[1]).to(
            img_feats.device
        )
        per_batch_features = []

        for batch_i, batch_bbox_feats in enumerate(bbox_feats):
            if not batch_bbox_feats:
                continue

            encoded_boxes = {"positive": None, "negative": None}

            # Process positive samples
            # positive_boxes = torch.cat(batch_bbox_feats["positive"], dim=1).to(
            #     img_feats.device
            # )
            if selected_class[batch_i] is not None:
                positive_normalized_distances, positive_encoded_boxes = (
                    self.encode_and_compute_distances(
                        img_feats[batch_i],
                        batch_bbox_feats["positive"],
                        enc_ind,
                        img_feats.device,
                        softmax,
                        cargs=cargs,
                        sel_cls=selected_class[batch_i],
                    )
                )
                attn_weights[batch_i, 1, :] = positive_normalized_distances
                encoded_boxes["positive"] = positive_encoded_boxes

            # Process negative samples
            # negative_boxes = torch.cat(batch_bbox_feats["negative"], dim=1).to(
            #     img_feats.device
            # )
            negative_normalized_distances, negative_encoded_boxes = (
                self.encode_and_compute_distances(
                    img_feats[batch_i],
                    batch_bbox_feats["negative"],
                    enc_ind,
                    img_feats.device,
                    softmax,
                    cargs=cargs,
                    sel_cls=0,
                )
            )
            attn_weights[batch_i, 0, :] = negative_normalized_distances
            encoded_boxes["negative"] = negative_encoded_boxes

            per_batch_features.append(encoded_boxes)

        # H, W = int(np.sqrt(img_feats.shape[1])), int(np.sqrt(img_feats.shape[1]))

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(
        #     attn_weights[0, 0].cpu().detach().reshape(H, W).numpy(),
        #     interpolation="nearest",
        # )
        # ax[0].axis("off")
        # ax[1].imshow(
        #     attn_weights[0, 1].cpu().detach().reshape(H, W).numpy(),
        #     interpolation="nearest",
        # )
        # ax[1].axis("off")
        # plt.savefig(f"plots/attn_weights_{enc_ind}.png")
        # plt.close()

        return attn_weights, per_batch_features

    def encode_bbox_features(self, bbox_features, scale_idx):
        """Encodes the bounding box features for a given scale."""
        encoded_boxes = []

        for bbox_feat in bbox_features:
            bbox = bbox_feat.unsqueeze(0)
            self.input_proj.eval()
            encoded_bbox = self.input_proj[scale_idx](bbox)
            self.input_proj.train()
            encoded_bbox = encoded_bbox.flatten(2).permute(0, 2, 1)
            encoded_boxes.append(encoded_bbox)

        return encoded_boxes

    def forward(
        self,
        img_feats,
        batch_bbox_feats=None,
        cargs=None,
        viz_attn=False,
        selected_class=None,
    ):

        assert len(img_feats) == len(self.in_channels)
        attn_weights = [None] * len(img_feats)
        bbox_features = [None] * len(img_feats)

        # 1.1. Input Projection of all image features
        proj_feats = [
            self.input_proj[i](feat).flatten(2).permute(0, 2, 1)
            for i, feat in enumerate(img_feats)
        ]  # len = 3, resnet outputs

        # for i, feat in enumerate(proj_feats):
        #     print(f"proj_feats[{i}]: {feat.shape}")

        # exit()

        # 1.2. Input Projection of all bbox features

        # batch_bbox_feats --> List per Batch -> Dict for positive and negative -> List of scales (3 per resnet) -> List of bbox features

        if cargs.interactive:
            projected_bbox_features = []

            for scale_idx, scale_features in enumerate(proj_feats):
                encoded_batch_features = []

                for batch_idx, batch_features in enumerate(batch_bbox_feats):
                    # Initialize dictionaries to store encoded features
                    encoded_boxes = {"positive": [], "negative": []}

                    # print(batch_features)

                    # Skip if there are no positive features in the current batch
                    if batch_features["positive"] == []:
                        continue

                    # Encode positive bounding box features
                    encoded_boxes["positive"] = self.encode_bbox_features(
                        batch_features["positive"][scale_idx], scale_idx
                    )

                    # Encode negative bounding box features
                    encoded_boxes["negative"] = self.encode_bbox_features(
                        batch_features["negative"][scale_idx], scale_idx
                    )

                    # Append encoded features of the current batch to the list
                    encoded_batch_features.append(encoded_boxes)

                # Append encoded features for the current scale to the main list
                projected_bbox_features.append(encoded_batch_features)

        # 1.3. Prepare positional embeddings
        pos_embeds = []
        for idx in range(len(proj_feats)):
            h = w = int(np.sqrt(proj_feats[idx].shape[1]))
            pos_embed = self.build_2d_sincos_position_embedding(
                w,
                h,
                self.hidden_dim,
                self.pe_temperature,
            )
            # print(f"pos_embed[{idx}]: {pos_embed.shape}")
            pos_embed = pos_embed.to(proj_feats[idx].device)
            pos_embeds.append(pos_embed)

        # 2.0. Interactive Cross Attention (Early Fusion)

        # for proj_feat in proj_feats:
        #     print(f"proj_feat: {proj_feat.shape}")

        if cargs.interactive and cargs.early_fusion:
            for enc_ind in range(len(proj_feats)):
                attn_w, bbox_f = self.interactive_cross_attention_heatmap(
                    img_feats=proj_feats[enc_ind],
                    bbox_feats=projected_bbox_features[enc_ind],
                    enc_ind=enc_ind,
                    pos_embed=pos_embeds[enc_ind],
                    softmax=cargs.softmax,
                    cargs=cargs,
                    selected_class=selected_class,
                )

                attn_weights[enc_ind] = attn_w
                bbox_features[enc_ind] = bbox_f

        # 2.2 General Encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):

                # print(f"Encoder {i} for scale {proj_feats[enc_ind].shape}...")

                memory, _ = self.encoder[i](
                    proj_feats[enc_ind], pos_embed=pos_embeds[enc_ind]
                )
                # memory_boxes, _ = self.encoder[i](bbox_proj_feats[enc_ind])

                proj_feats[enc_ind] = memory

        # 2.3 Interactive Cross Attention (Late Fusion)
        if cargs.interactive and cargs.late_fusion:
            for enc_ind in range(len(proj_feats)):
                attn_w, bbox_f = self.interactive_cross_attention_heatmap(
                    img_feats=proj_feats[enc_ind],
                    bbox_feats=projected_bbox_features[enc_ind],
                    enc_ind=enc_ind,
                    pos_embed=pos_embeds[enc_ind],
                    softmax=cargs.softmax,
                    cargs=cargs,
                    selected_class=selected_class,
                )

                attn_weights[enc_ind] = attn_w
                bbox_features[enc_ind] = bbox_f

        # B = proj_feats[0].shape[0]

        # fig, ax = plt.subplots(B, len(attn_weights), figsize=(10, 3 * B))

        # for i, attn in enumerate(attn_weights):
        #     B, C, S = attn.shape
        #     h = w = int(np.sqrt(S))

        #     attn = attn.reshape(B, C, h, w)
        #     for j in range(B):
        #         ax[j, i].imshow(
        #             attn[j, 1].cpu().detach().numpy(),
        #             interpolation="nearest",
        #         )
        #         ax[j, i].axis("off")

        # plt.tight_layout()
        # plt.savefig("plots/attn_weights_pos_2.png")

        # print("Before", [(attn.shape, attn.min(), attn.max()) for attn in attn_weights])

        # exit()

        for i, attn in enumerate(attn_weights):
            B, C, S = attn.shape

            h = w = int(np.sqrt(S))

            attn = attn.reshape(B, C, h, w)

            # attn = self.attn_conv[i](attn)

            attn_weights[i] = attn

            if cargs.norm_batch:
                attn_weights[i] = self.norm_batch[i](attn_weights[i])

        # 2.4 Attn Convolution
        if cargs.attn_fusion:

            # print([attn.shape for attn in attn_weights])

            # broadcasting and fusion
            inner_outs_attn = [attn_weights[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_heigh = inner_outs_attn[0]
                feat_low = attn_weights[idx - 1]

                feat_heigh = self.lateral_convs_attn[len(self.in_channels) - 1 - idx](
                    feat_heigh
                )

                inner_outs_attn[0] = feat_heigh
                upsample_feat = F.interpolate(
                    feat_heigh, scale_factor=2.0, mode="nearest"
                )
                inner_out = self.fpn_blocks_attn[len(self.in_channels) - 1 - idx](
                    torch.concat([upsample_feat, feat_low], dim=1)
                )
                inner_outs_attn.insert(0, inner_out)

            output_features_attn = [inner_outs_attn[0]]
            for idx in range(len(self.in_channels) - 1):
                feat_low = output_features_attn[-1]
                feat_height = inner_outs_attn[idx + 1]
                downsample_feat = self.downsample_convs_attn[idx](feat_low)

                out = self.pan_blocks_attn[idx](
                    torch.concat([downsample_feat, feat_height], dim=1)
                )
                output_features_attn.append(out)

            for i, attn in enumerate(output_features_attn):

                attn = self.attn_conv[i](attn)

                attn_weights[i] = attn.flatten(2).permute(0, 2, 1)

        elif cargs.attn_conv:

            for i, attn in enumerate(attn_weights):

                # # if cargs.fancy_plots:
                # for batch_i in range(B):
                #     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                #     ax.imshow(
                #         attn[batch_i, 1].cpu().detach().numpy(), interpolation="nearest"
                #     )
                #     ax.axis("off")
                #     plt.savefig(f"plots/attn_weights_{i}_b{batch_i}.png")
                #     plt.close()

                #     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                #     ax.imshow(
                #         attn[batch_i, 0].cpu().detach().numpy(), interpolation="nearest"
                #     )
                #     ax.axis("off")
                #     plt.savefig(f"plots/attn_weights_{i}_b{batch_i}_neg.png")
                #     plt.close()

                #     break

                attn = self.attn_conv[i](attn)
                attn_weights[i] = attn.flatten(2).permute(0, 2, 1)

            # exit()

        else:
            for i, attn in enumerate(attn_weights):

                attn_weights[i] = attn.flatten(2).permute(0, 2, 1)[:, :, 1:]

        attn_weights = [F.relu(attn) for attn in attn_weights]

        if self.lump_edition:
            output_features = attn_weights.copy()

            for i, feat in enumerate(output_features):
                h = w = int(np.sqrt(feat.shape[1]))
                output_features[i] = (
                    feat.permute(0, 2, 1).reshape(-1, 1, h, w).contiguous()
                )

            return output_features, attn_weights, bbox_features, proj_feats

        if cargs.interactive and not cargs.late_combine:
            for i, attn in enumerate(attn_weights):
                feats = proj_feats[i] * attn
                if cargs.residual:
                    feats = proj_feats[i] + feats
                proj_feats[i] = F.relu(feats)

        # 2.4 Reshape to image
        for i, feat in enumerate(proj_feats):
            h = w = int(np.sqrt(feat.shape[1]))
            proj_feats[i] = (
                feat.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
            )

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)

            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        output_features = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = output_features[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.concat([downsample_feat, feat_height], dim=1)
            )
            output_features.append(out)

        return output_features, attn_weights, bbox_features, proj_feats
