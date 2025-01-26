# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.ops import resize
from mmseg.utils import get_root_logger

# from mmseg.models.builder import BACKBONES
from mmseg.models.builder import BACKBONES
# from mmcv.utils import Registry
from mmseg.models.utils import PatchEmbed
# from ..utils import PatchEmbed
import time
from typing import Callable, Tuple

# from mmcv.utils import Registry

# MODELS = Registry('models')

# Reuse the same instance elsewhere
# PIPELINES = Registry('pipeline')



def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """

    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x, unm_idx, src_idx, dst_idx = merge(x * size)
    size, unm_idx, src_idx, dst_idx = merge(size)

    x = x / size

    return x, size, unm_idx, src_idx, dst_idx

def do_not_reduce(x, mode=None):
    return x, None, None, None

def do_not_reconstruct(x, unm_idx, src_idx, dst_idx, out_layer):
    return x

def custom_merge(
    metric: torch.Tensor,
    token_history,
    token_history_processed,
    r: float,
    class_token: bool = False,
    distill_token:bool = False
) -> tuple[Callable, Callable]:

    if r >= 1:
        return do_not_reduce, do_not_reconstruct
        
    if token_history.dim() == 0:
        return do_not_reduce, do_not_reconstruct

    with torch.no_grad():
        # If there is no token history, no merge can happen
        if token_history is None:
            return do_not_merge, do_not_unmerge
        
        a = metric        # tokens from the image, Shape: (1, x, 1024)
        b = token_history # tokens from database,  Shape(1, x, 1024)
        
        a_norm = a.norm(dim=-1, keepdim=True)  # Length of each token as vector, Shape: (1, x, 1)
        b_norm = b.norm(dim=-1, keepdim=True)  # Length of each token as vector, Shape: (1, x, 1)
        
        dot_product  = a @ b.transpose(-1, -2)           # above cosine similarity
        norm_product = a_norm @ b_norm.transpose(-1, -2) # below cosine similarity
        
        # Introduce epsilon to prevent division by 0
        epsilon = 1e-8 
        norm_product = norm_product + epsilon
        
        # Cosine Similarity
        cosine_similarity = dot_product / norm_product
        
        # Find max tokens from database for each token from image
        node_max, node_idx = cosine_similarity.max(dim=-1)
        
        # apply maks to tokens exceeding threshold
        mask = node_max > r
        
        if torch.any(mask):
            # Token Reduction
            src_idx = torch.where(mask)[1]
            src_idx = src_idx.unsqueeze(0).unsqueeze(-1)
            
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        
        else:
            # No Token Reduction
            src_idx = None
            dst_idx = None
        
        unm_idx = torch.where(~mask)[1]
        unm_idx = unm_idx.unsqueeze(0).unsqueeze(-1)



    def reduce(x: torch.Tensor, mode=None) -> torch.Tensor:
        
        if(src_idx is not None): # If there were tokens matched
            
            src = x # the source of tokens is just x
    
            n, t1, c = src.shape
    
            _, t2, _ = unm_idx.shape

            # return all tokens that haven't been matched, this becomes the 'reduced token set'
            unmatched_tokens = src.gather(dim=-2, index=unm_idx.expand(n, t2, c))
        else:
            # no matches, no reduction
            unmatched_tokens = x
        return unmatched_tokens, unm_idx, src_idx, dst_idx

    def reconstruct(x: torch.Tensor, unm_idx: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor, out_layer):
        if src_idx is not None: # If there were tokens matched

            unm = x # origianl tensor to be extended
            dst = token_history_processed[out_layer] # processed equivalents of the tokens matched
        
            n, unm_len, c = unm.shape  # Shape of the input
            r = dst_idx.size(1)        # Reduction size
            total_len = unm_len + r    # Length of the output
        
            # Gather the src tokens from `dst` at positions specified by `dst_idx`
            src = torch.index_select(dst, dim=1, index=dst_idx.view(-1)).view(n, r, c)
        
            # Concatenate the full unm tokens and the source tokens
            # This tensor has the size of the output tensor needed
            combined = torch.cat((unm, src), dim=1)
        
            # Concatenate the index tensors.
            indices = torch.cat((unm_idx, src_idx), dim=1).expand(n, total_len, c)
        
            # Use the indeces to fill in the full tokens in the right place
            out = torch.zeros_like(combined, device=x.device).scatter_(
                dim=1, index=indices, src=combined
            )
        
            return out
        else: # No reconstruction required
            return x

    return reduce, reconstruct



class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False,
                 layer=0,
                 device='cuda:0'):
        super(TransformerEncoderLayer, self).__init__()

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
        self.with_cp = with_cp

        # my additions
        self.layer = layer
        self.unmerge_fn = None
        self.unm_idx = None
        self.src_idx = None
        self.dst_idx = None
        self.device = device

        self.token_history = []

        self.token_history_processed = {}

        self.token_history_tensor = torch.zeros([]).to(self.device)
        self.token_history_count_tensor = []

        

        self.token_history_processed_circular = {}
        self.token_history_circular = torch.zeros((1,2048, 1024), device=self.device)
        self.max_len = 2048
        self.token_size = 1024
        self.index_token_history = 0
        self.index_token_history_processed = {}

        self.r = 0.95
        self.reduction_interval = 8

        for out_layer in [7, 15, 23]:
            self.token_history_processed.setdefault(out_layer, torch.zeros((1,2048, 1024), device=self.device))
            self.index_token_history_processed[out_layer] = 0

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



    def forward(self, x):

        def _inner_forward(x):
            # self attention layer
            x = self.attn(self.norm1(x), identity=x)

            # if layer is one that should be reduced
            if self.layer % self.reduction_interval == 0:
                _, n_tokens, _ = x.shape

                # if len(self.token_history_count_tensor)>0:
                # stacked_history = 
                # stacked_processed_history = 
                merge, unmerge = custom_merge(x, self.token_history_circular , self.token_history_processed, r=self.r)
                
                x, size, unm_idx, src_idx, dst_idx = merge_wavg(merge, x)

                self.unmerge_fn = unmerge
                self.unm_idx = unm_idx
                self.src_idx = src_idx
                self.dst_idx = dst_idx
                
                _, batch_size, token_size = x.shape
                end_idx = self.index_token_history + batch_size
                if end_idx <= self.max_len:
                    # Add tokens without wrapping
                    self.token_history_circular[0, self.index_token_history:end_idx, :] = x[0]
                else:
                    # Wrap around: split the tokens and write in two parts
                    overflow = end_idx - self.max_len
                    self.token_history_circular[0, self.index_token_history:self.max_len, :] = x[0, :batch_size - overflow, :]
                    self.token_history_circular[0, :overflow, :] = x[0, batch_size - overflow:, :]
        
                # Update the index, wrapping around
                self.index_token_history = end_idx % self.max_len

            # I add myself to the token merge
            x = self.ffn(self.norm2(x), identity=x)

            return x

        if self.with_cp and x.requires_grad:

            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x

    def unmerge(self, x, unm_idx, src_idx, dst_idx, out_layer):

        if self.layer % self.reduction_interval == 0 or unm_idx is not None:

            reconstructed_tokens = self.unmerge_fn(x, unm_idx, src_idx, dst_idx, out_layer)

            _, batch_size, token_size = x.shape
            end_idx = self.index_token_history_processed[out_layer] + batch_size
            
            if end_idx <= self.max_len:
                # Add tokens without wrapping
                self.token_history_processed[out_layer][0, self.index_token_history_processed[out_layer]:end_idx, :] = x[0]
            else:
                # Wrap around: split the tokens and write in two parts
                overflow = end_idx - self.max_len
                self.token_history_processed[out_layer][0, self.index_token_history_processed[out_layer]:self.max_len, :] = x[0, :batch_size - overflow, :]
                self.token_history_processed[out_layer][0, :overflow, :] = x[0, batch_size - overflow:, :]

            # Update the index, wrapping around if needed
            self.index_token_history_processed[out_layer] = end_idx % self.max_len

            return reconstructed_tokens
        else:
            return x


# Check if the model is already registered
if 'TokenReducingVisionTransformer' not in BACKBONES:

    @BACKBONES.register_module()
    class TokenReducingVisionTransformer(BaseModule):
        """Vision Transformer.
    
        This backbone is the implementation of `An Image is Worth 16x16 Words:
        Transformers for Image Recognition at
        Scale <https://arxiv.org/abs/2010.11929>`_.
    
        Args:
            img_size (int | tuple): Input image size. Default: 224.
            patch_size (int): The patch size. Default: 16.
            in_channels (int): Number of input channels. Default: 3.
            embed_dims (int): embedding dimension. Default: 768.
            num_layers (int): depth of transformer. Default: 12.
            num_heads (int): number of attention heads. Default: 12.
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
                Default: 4.
            out_indices (list | tuple | int): Output from which stages.
                Default: -1.
            qkv_bias (bool): enable bias for qkv if True. Default: True.
            drop_rate (float): Probability of an element to be zeroed.
                Default 0.0
            attn_drop_rate (float): The drop out rate for attention layer.
                Default 0.0
            drop_path_rate (float): stochastic depth rate. Default 0.0
            with_cls_token (bool): Whether concatenating class token into image
                tokens as transformer input. Default: True.
            output_cls_token (bool): Whether output the cls_token. If set True,
                `with_cls_token` must be True. Default: False.
            norm_cfg (dict): Config dict for normalization layer.
                Default: dict(type='LN')
            act_cfg (dict): The activation config for FFNs.
                Default: dict(type='GELU').
            patch_norm (bool): Whether to add a norm in PatchEmbed Block.
                Default: False.
            final_norm (bool): Whether to add a additional layer to normalize
                final feature map. Default: False.
            interpolate_mode (str): Select the interpolate mode for position
                embeding vector resize. Default: bicubic.
            num_fcs (int): The number of fully-connected layers for FFNs.
                Default: 2.
            norm_eval (bool): Whether to set norm layers to eval mode, namely,
                freeze running stats (mean and var). Note: Effect on Batch Norm
                and its variants only. Default: False.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save
                some memory while slowing down the training speed. Default: False.
            pretrained (str, optional): model pretrained path. Default: None.
            init_cfg (dict or list[dict], optional): Initialization config dict.
                Default: None.
        """
    
        def __init__(self,
                     img_size=224,
                     patch_size=16,
                     in_channels=3,
                     embed_dims=768,
                     num_layers=12,
                     num_heads=12,
                     mlp_ratio=4,
                     out_indices=-1,
                     qkv_bias=True,
                     drop_rate=0.,
                     attn_drop_rate=0.,
                     drop_path_rate=0.,
                     with_cls_token=True,
                     output_cls_token=False,
                     norm_cfg=dict(type='LN'),
                     act_cfg=dict(type='GELU'),
                     patch_norm=False,
                     final_norm=False,
                     interpolate_mode='bicubic',
                     num_fcs=2,
                     norm_eval=False,
                     with_cp=False,
                     pretrained=None,
                     init_cfg=None,
                     device='cuda:0'):
            super(TokenReducingVisionTransformer, self).__init__(init_cfg=init_cfg)
    
            self.reconstruction_times = []
            self.device = device
    
            if isinstance(img_size, int):
                img_size = to_2tuple(img_size)
            elif isinstance(img_size, tuple):
                if len(img_size) == 1:
                    img_size = to_2tuple(img_size[0])
                assert len(img_size) == 2, \
                    f'The size of image should have length 1 or 2, ' \
                    f'but got {len(img_size)}'
    
            if output_cls_token:
                assert with_cls_token is True, f'with_cls_token must be True if' \
                    f'set output_cls_token to True, but got {with_cls_token}'
    
            assert not (init_cfg and pretrained), \
                'init_cfg and pretrained cannot be set at the same time'
            if isinstance(pretrained, str):
                warnings.warn('DeprecationWarning: pretrained is deprecated, '
                              'please use "init_cfg" instead')
                self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
            elif pretrained is not None:
                raise TypeError('pretrained must be a str or None')
    
            self.img_size = img_size
            self.patch_size = patch_size
            self.interpolate_mode = interpolate_mode
            self.norm_eval = norm_eval
            self.with_cp = with_cp
            self.pretrained = pretrained
    
            self.patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims,
                conv_type='Conv2d',
                kernel_size=patch_size,
                stride=patch_size,
                padding='corner',
                norm_cfg=norm_cfg if patch_norm else None,
                init_cfg=None,
            )
    
            num_patches = (img_size[0] // patch_size) * \
                (img_size[1] // patch_size)
    
            self.with_cls_token = with_cls_token
            self.output_cls_token = output_cls_token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dims))
            self.drop_after_pos = nn.Dropout(p=drop_rate)
    
            if isinstance(out_indices, int):
                if out_indices == -1:
                    out_indices = num_layers - 1
                self.out_indices = [out_indices]
            elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
                self.out_indices = out_indices
            else:
                raise TypeError('out_indices must be type of int, list or tuple')
    
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
            ]  # stochastic depth decay rule
    
            self.layers = ModuleList()
            for i in range(num_layers):
                self.layers.append(
                    TransformerEncoderLayer(
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        feedforward_channels=mlp_ratio * embed_dims,
                        attn_drop_rate=attn_drop_rate,
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[i],
                        num_fcs=num_fcs,
                        qkv_bias=qkv_bias,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        with_cp=with_cp,
                        batch_first=True,
                        layer=i,
                        device=self.device))
    
            self.final_norm = final_norm
            if final_norm:
                self.norm1_name, norm1 = build_norm_layer(
                    norm_cfg, embed_dims, postfix=1)
                self.add_module(self.norm1_name, norm1)
    
            self.history = torch.tensor([]).to(self.device)
    
        @property
        def norm1(self):
            return getattr(self, self.norm1_name)
    
        def init_weights(self):
            if (isinstance(self.init_cfg, dict)
                    and self.init_cfg.get('type') == 'Pretrained'):
                logger = get_root_logger()
                checkpoint = CheckpointLoader.load_checkpoint(
                    self.init_cfg['checkpoint'], logger=logger, map_location=self.device)
    
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
    
                if 'pos_embed' in state_dict.keys():
                    if self.pos_embed.shape != state_dict['pos_embed'].shape:
                        logger.info(msg=f'Resize the pos_embed shape from '
                                    f'{state_dict["pos_embed"].shape} to '
                                    f'{self.pos_embed.shape}')
                        h, w = self.img_size
                        pos_size = int(
                            math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                        state_dict['pos_embed'] = self.resize_pos_embed(
                            state_dict['pos_embed'],
                            (h // self.patch_size, w // self.patch_size),
                            (pos_size, pos_size), self.interpolate_mode)
    
                load_state_dict(self, state_dict, strict=False, logger=logger)
            elif self.init_cfg is not None:
                super(TokenReducingVisionTransformer, self).init_weights()
            else:
                # We only implement the 'jax_impl' initialization implemented at
                # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
                trunc_normal_(self.pos_embed, std=.02)
                trunc_normal_(self.cls_token, std=.02)
                for n, m in self.named_modules():
                    if isinstance(m, nn.Linear):
                        trunc_normal_(m.weight, std=.02)
                        if m.bias is not None:
                            if 'ffn' in n:
                                nn.init.normal_(m.bias, mean=0., std=1e-6)
                            else:
                                nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Conv2d):
                        kaiming_init(m, mode='fan_in', bias=0.)
                    elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                        constant_init(m, val=1.0, bias=0.)
    
        def _pos_embeding(self, patched_img, hw_shape, pos_embed):
            """Positioning embeding method.
    
            Resize the pos_embed, if the input image size doesn't match
                the training size.
            Args:
                patched_img (torch.Tensor): The patched image, it should be
                    shape of [B, L1, C].
                hw_shape (tuple): The downsampled image resolution.
                pos_embed (torch.Tensor): The pos_embed weighs, it should be
                    shape of [B, L2, c].
            Return:
                torch.Tensor: The pos encoded image feature.
            """
            assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
                'the shapes of patched_img and pos_embed must be [B, L, C]'
            x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
            if x_len != pos_len:
                if pos_len == (self.img_size[0] // self.patch_size) * (
                        self.img_size[1] // self.patch_size) + 1:
                    pos_h = self.img_size[0] // self.patch_size
                    pos_w = self.img_size[1] // self.patch_size
                else:
                    raise ValueError(
                        'Unexpected shape of pos_embed, got {}.'.format(
                            pos_embed.shape))
                pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                                  (pos_h, pos_w),
                                                  self.interpolate_mode)
            return self.drop_after_pos(patched_img + pos_embed)
    
        @staticmethod
        def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
            """Resize pos_embed weights.
    
            Resize pos_embed using bicubic interpolate method.
            Args:
                pos_embed (torch.Tensor): Position embedding weights.
                input_shpae (tuple): Tuple for (downsampled input image height,
                    downsampled input image width).
                pos_shape (tuple): The resolution of downsampled origin training
                    image.
                mode (str): Algorithm used for upsampling:
                    ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                    ``'trilinear'``. Default: ``'nearest'``
            Return:
                torch.Tensor: The resized pos_embed of shape [B, L_new, C]
            """
            assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
            pos_h, pos_w = pos_shape
            # keep dim for easy deployment
            cls_token_weight = pos_embed[:, 0:1]
            pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
            pos_embed_weight = pos_embed_weight.reshape(
                1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
            pos_embed_weight = resize(
                pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
            pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
            pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
            return pos_embed
    
        def forward(self, inputs):
            B = inputs.shape[0]
    
            x, hw_shape = self.patch_embed(inputs)
    
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = self._pos_embeding(x, hw_shape, self.pos_embed)
            x = x[:, 1:]
    
            outs_tensor = []
            for i, layer in enumerate(self.layers):
    
                x = layer(x)
                
                if i == len(self.layers) - 1:
                    if self.final_norm:
                        x = self.norm1(x)
                
                if i in self.out_indices:
                    # print('out_indices', self.out_indices)
    
                    x_unmerged = x
    
                    for j, layer in reversed(list(enumerate(self.layers[:i+1]))):
                        x_unmerged = layer.unmerge(x_unmerged, layer.unm_idx, layer.src_idx, layer. dst_idx, i)
      
                    out = x_unmerged
                    
                    B, _, C = out.shape
    
                    out = out.reshape(B, hw_shape[0], hw_shape[1],
                                      C).permute(0, 3, 1, 2).contiguous()
                    
                    out_tensor = torch.tensor(out)
                    outs_tensor.append(out_tensor)
    
                    # print('out tensor shape', out_tensor.shape)
    
            outs_tensor = torch.stack(outs_tensor)
            
            return outs_tensor
    
        def train(self, mode=True):
            super(TokenReducingVisionTransformer, self).train(mode)
            if mode and self.norm_eval:
                for m in self.modules():
                    if isinstance(m, nn.LayerNorm):
                        m.eval()
