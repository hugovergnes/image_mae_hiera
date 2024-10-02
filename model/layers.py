from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from model.hiera_utils import do_pool


class MultiScaleAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask
    units. See `Unroll` for more details.
    """

    def __init__(
        self,
        dim,
        dim_out,
        heads,
        q_stride=1,
        window_size=0,
        use_mask_unit_attn=False,
    ):
        """
        Initialize the model.

        Args:
            dim (int): The input feature dimension.
            dim_out (int): The output feature dimension.
            heads (int): The number of attention heads.
            q_stride (int, optional): The stride for pooling the query. Defaults to 1.
            window_size (int, optional): The current (flattened) size of a mask unit
                after pooling (if any). Defaults to 0.
            use_mask_unit_attn (bool, optional): Whether to use Mask Unit or Global
                Attention. Defaults to False.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x):
        """Input should be of shape [batch, tokens, channels]."""
        B, N, _ = x.shape
        num_windows = (
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        q_stride=1,
        window_size=0,
        use_mask_unit_attn=False,
    ):
        """_summary_

        Args:
            dim (int): _description_
            dim_out (int): _description_
            heads (int): _description_
            mlp_ratio (float, optional): _description_. Defaults to 4.0.
            drop_path (float, optional): _description_. Defaults to 0.0.
            norm_layer (nn.Module, optional): _description_. Defaults to nn.LayerNorm.
            act_layer (nn.Module, optional): _description_. Defaults to nn.GELU.
            q_stride (int, optional): _description_. Defaults to 1.
            window_size (int, optional): _description_. Defaults to 0.
            use_mask_unit_attn (bool, optional): _description_. Defaults to False.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        activation=nn.GELU,
    ):
        """_summary_

        Args:
            input_dim (int): _description_
            hidden_dim (int): _description_
            output_dim (int): _description_
            num_layers (int): _description_
            activation (nn.Module, optional): _description_. Defaults to nn.GELU.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
