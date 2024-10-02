import pytest
import torch

from model.hiera_mae import MaskedAutoencoderHiera


class TestHiera:
    @pytest.fixture
    def mock_input_tensor(self):
        return torch.randn(3, 1, 1280, 768)

    def test_get_pos_embed(self):
        model = MaskedAutoencoderHiera(
            input_size=(1280, 768),
            in_chans=1,
            patch_kernel=(4, 4),
            patch_stride=(4, 4),
            use_sin_cos_pos_embed=False,
        )
        assert model._get_pos_embed((320, 192)).shape == (1, 320, 192, 96)

        model = MaskedAutoencoderHiera(
            input_size=(1280, 768),
            in_chans=1,
            patch_kernel=(4, 4),
            patch_stride=(4, 4),
            use_sin_cos_pos_embed=True,
        )
        assert model._get_pos_embed((320, 192)).shape == (1, 320, 192, 96)

    def test_forward(self, mock_input_tensor):
        model = MaskedAutoencoderHiera(
            input_size=(1280, 768),
            in_chans=1,
            patch_kernel=(4, 4),
            patch_stride=(4, 4),
            mask_ratio=0.75,
        )
        patched, pred, pred_mask = model(mock_input_tensor)
        assert patched.shape == (3, 960, 1024)

        model = MaskedAutoencoderHiera(
            input_size=(1280, 768),
            in_chans=1,
            patch_kernel=(4, 4),
            patch_stride=(4, 4),
            mask_ratio=0.75,
            use_sin_cos_pos_embed=True,
        )
        patched, pred, pred_mask = model(mock_input_tensor)
        assert patched.shape == (3, 960, 1024)
