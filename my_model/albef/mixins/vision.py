from functools import partial
import torch
from torch import nn
from my_model.vit import VisionTransformer

class VisionBuilderMixin:
    def _build_vit(self, img_size):
        return VisionTransformer(
            img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )


