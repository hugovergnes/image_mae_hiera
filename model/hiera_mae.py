from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import MultiScaleBlock
from model.patch_embed import PatchEmbed

from model.hiera_utils import undo_windowing, Unroll, Reroll

from model.pos_embed import get_2d_sincos_pos_embed


def apply_fusion_head(head, x):
    B, num_mask_units = x.shape[0:2]
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(0, 3, 1, 2))
    x = x.permute(0, 2, 3, 1).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])
    return x


class MaskedAutoencoderHiera(nn.Module):
    def __init__(
        self,
        input_size=(1280, 768),
        in_chans=3,
        embed_dim=96,
        patch_kernel=(4, 4),
        patch_stride=(4, 4),
        patch_padding=(0, 0),
        num_heads=1,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=0.75,
        stages=(2, 3, 16, 3),
        q_pool=3,
        q_stride=(2, 2),
        mask_unit_size=(8, 8),
        mask_unit_attn=(True, True, False, False),
        dim_mul=2.0,
        head_mul=2.0,
        drop_path_rate=0.0,
        window_pos_embed_bkg_spatial_size=(14, 14),
        use_sin_cos_pos_embed=False,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=16,
    ):
        """_summary_

        Args:
            input_size (tuple, optional): _description_. Defaults to (1280, 768).
            in_chans (int, optional): _description_. Defaults to 3.
            embed_dim (int, optional): _description_. Defaults to 96.
            patch_kernel (tuple, optional): _description_. Defaults to (4, 4).
            patch_stride (tuple, optional): _description_. Defaults to (4, 4).
            patch_padding (tuple, optional): _description_. Defaults to (0, 0).
            num_heads (int, optional): _description_. Defaults to 1.
            mlp_ratio (float, optional): _description_. Defaults to 4.0.
            norm_layer (_type_, optional): _description_. Defaults to partial(nn.LayerNorm, eps=1e-6).
            mask_ratio (float, optional): _description_. Defaults to 0.75.
            stages (tuple, optional): _description_. Defaults to (2, 3, 16, 3).
            q_pool (int, optional): _description_. Defaults to 3.
            q_stride (tuple, optional): _description_. Defaults to (2, 2).
            mask_unit_size (tuple, optional): _description_. Defaults to (8, 8).
            mask_unit_attn (tuple, optional): _description_. Defaults to (True, True, False, False).
            dim_mul (float, optional): _description_. Defaults to 2.0.
            head_mul (float, optional): _description_. Defaults to 2.0.
            drop_path_rate (float, optional): _description_. Defaults to 0.0.
            window_pos_embed_bkg_spatial_size (tuple, optional): _description_. Defaults to (14, 14).
            use_sin_cos_pos_embed (bool, optional): _description_. Defaults to False.
            decoder_embed_dim (int, optional): _description_. Defaults to 512.
            decoder_depth (int, optional): _description_. Defaults to 2.
            decoder_num_heads (int, optional): _description_. Defaults to 16.
        """
        super().__init__()

        self.patch_embed = PatchEmbed(
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            in_chans=in_chans,
            embed_dim=embed_dim,
            img_size=input_size,
        )

        depth = sum(stages)
        self.patch_stride = patch_stride
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)

        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.patch_embed.grid_size, self.mask_unit_size)
        ]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

        # Set up pos embed
        self.use_sin_cos_pos_embed = use_sin_cos_pos_embed
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.set_up_pos_embed(embed_dim)
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, *mask_unit_size))

        # Setup roll and reroll modules
        self.unroll = Unroll(
            input_size, patch_stride, [q_stride] * len(self.stage_ends[:-1])
        )
        self.reroll = Reroll(
            input_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.mask_ratio = mask_ratio

        encoder_dim_out = self.blocks[-1].dim_out
        self.encoder_norm = norm_layer(encoder_dim_out)
        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.patch_embed.grid_size, self.q_stride)
        ]

        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for i in self.stage_ends[: self.q_pool]:
            kernel = [
                i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)
            ]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            self.multi_scale_fusion_heads.append(
                nn.Conv2d(
                    self.blocks[i].dim_out,
                    encoder_dim_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )

        # final stage, no transform
        self.multi_scale_fusion_heads.append(nn.Identity())

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim
            ),
            requires_grad=not self.use_sin_cos_pos_embed,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                MultiScaleBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (self.q_stride[-1] ** self.q_pool)

        self.decoder_pred = nn.Linear(decoder_embed_dim, self.pred_stride**2 * in_chans)
        # --------------------------------------------------------------------------

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.use_sin_cos_pos_embed:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=False
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1],
                self.tokens_spatial_shape_final,
                cls_token=False,
            )
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
            )
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def _mae_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_up_pos_embed(self, embed_dim):
        if not self.use_sin_cos_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim),
                requires_grad=False,
            )

    def _get_pos_embed(self, hw):
        """_summary_

        Args:
            hw (Tuple[int, int]): _description_

        Returns:
            torch.Tensor: _description_
        """
        h, w = hw

        if self.use_sin_cos_pos_embed:
            pos_embed = self.pos_embed.view(
                self.pos_embed.shape[0], h, w, self.pos_embed.shape[-1]
            )
            pos_embed = pos_embed.permute(0, 3, 1, 2)
        else:
            pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")

        window_embed = self.pos_embed_window
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    @staticmethod
    def patchify(x, p):
        """_summary_

        Args:
            x (_type_): _description_
            p (int): stride to make square patches.

        Returns:
            _type_: _description_
        """
        B, _, H, W = x.shape
        h, w = H // p, W // p

        patched_image = x.reshape(shape=(x.shape[0], 1, h, p, w, p))
        patched_image = torch.einsum("nchpwq->nhwpqc", patched_image)
        return patched_image.reshape(shape=(B, h * w, p**2))  # B, L, P**2

    def unpatchify(self, x):
        """
        x: (B, L, patch_size**2)
        imgs: (B, 1, H, W)
        """
        p = self.pred_stride
        h, w = self.input_size[0] // p, self.input_size[1] // p

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return imgs

    def unravel_mask(self, mask):
        # unravel mask to visualize. B, L -> B, H, W, 1
        x = mask.detach()
        x = x.unsqueeze(-1).repeat(1, 1, self.pred_stride**2)  # (N, H*W, p*p)
        x = self.unpatchify(x)
        return torch.einsum("nchw->nhwc", x)

    def get_random_mask(self, x, mask_ratio):
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def forward_encoder(self, x):
        """This skips the forward in the Hiera class

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])

        mask = self.get_random_mask(x, self.mask_ratio)

        # Flatten
        x = x.view(x.shape[0], -1, x.shape[-1])

        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )

        intermediates = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))

        # x may not always be in spatial order here.
        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and
        # q_stride = (2, 2), not all unrolls were consumed,
        # intermediates[-1] is x in spatial order

        # Resolution unchanged after q_pool stages, so skip those features
        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        # Multi-scale fusion
        x = 0.0
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            x += apply_fusion_head(head, interm_x)

        x = self.encoder_norm(x)

        return x, mask

    def forward_decoder(self, x, mask):
        """_summary_

        Args:
            x (_type_): _description_
            mask (_type_): _description_

        Returns:
            Tuple: Prediction and Mask. Prediction has shape
                B, NMUs, self.pred_stride**2 * in_chans
        """
        # Embed tokens
        x = self.decoder_embed(x)

        # Combine visible and mask tokens

        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, #MUs_all]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return x, mask

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): _description_
            mask_ratio (float, optional): _description_. Defaults to 0.6.

        Returns:
            Tuple: _description_
        """
        patched = self.patchify(x, self.pred_stride)

        latent, mask = self.forward_encoder(x)
        pred, pred_mask = self.forward_decoder(latent, mask)

        return patched, pred, ~pred_mask


class MAEHieraSmall(MaskedAutoencoderHiera):
    def __init__(
        self,
        stages=[1, 2, 11, 2],
        **kwargs,
    ):
        super(MAEHieraSmall, self).__init__(
            stages=stages,
            **kwargs,
        )


class MAEHieraBasePlus(MaskedAutoencoderHiera):
    def __init__(
        self,
        embed_dim=112,
        num_heads=2,
        **kwargs,
    ):
        super(MAEHieraBasePlus, self).__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            **kwargs,
        )
