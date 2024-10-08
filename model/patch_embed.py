import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CubeEmbed(nn.Module):
    """
    Video to Cube Embedding.
    """

    def __init__(
        self,
        kernel_size=(2, 4, 4),
        stride=(2, 4, 4),
        padding=(0, 0, 0),
        in_chans=3,
        embed_dim=768,
        img_size=None,
    ):
        """_summary_

        Args:
            kernel_size (Tuple): kernel size of the projection layer.
                Defaults to (7, 7).
            stride (Tuple): stride of the projection layer. Defaults to (4, 4).
            padding (Tuple): padding size of the projection layer. Defaults to (3, 3).
            in_chans (int): Number of input image channels. Defaults to 3.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
                Defaults to 768.
            img_size (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        if img_size is not None:
            grid_size = (np.array(img_size) + np.array(padding)) / np.array(stride)
            self.grid_size = grid_size.astype(int)
            self.num_patches = int(self.grid_size.prod())

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: Output tensor has shape B, H_tokens, W_tokens, C
        """
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
        in_chans=3,
        embed_dim=768,
        img_size=None,
    ):
        """_summary_

        Args:
            kernel_size (Tuple): kernel size of the projection layer.
                Defaults to (7, 7).
            stride (Tuple): stride of the projection layer. Defaults to (4, 4).
            padding (Tuple): padding size of the projection layer. Defaults to (3, 3).
            in_chans (int): Number of input image channels. Defaults to 3.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
                Defaults to 768.
            img_size (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        if img_size is not None:
            grid_size = (np.array(img_size) + np.array(padding)) / np.array(stride)
            self.grid_size = grid_size.astype(int)
            self.num_patches = int(self.grid_size.prod())

    def forward(self, x, mask=None):
        """_summary_

        Args:
            x (torch.Tensor): _description_
            mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: Output tensor has shape B, H_tokens, W_tokens, C
        """
        if mask is None:
            x = self.proj(x)
        else:
            interp_mask = F.interpolate(mask.float(), size=x.shape[2:], mode="nearest")
            x = self.proj(x * interp_mask.bool())
        return x.permute(0, 2, 3, 1)
