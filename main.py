import torch
from torch.autograd import profiler

from model.hiera_mae import MaskedAutoencoderHiera


class MAELoss(torch.nn.Module):
    def __init__(self, norm_pix_loss=True):
        super(MAELoss, self).__init__()
        self.norm_pix_loss = norm_pix_loss

    def __call__(self, image, predicted_image, mask):
        """_summary_

        Args:
            image (torch.Tensor): _description_
            predicted_image (torch.Tensor): _description_
            mask (torch.Tensor): Mask at the prediction resolution. 1 is dropped,
                0 is kept.

        Returns:
            _type_: _description_
        """
        if self.norm_pix_loss:
            mu = image.mean(dim=-1, keepdim=True)
            sigma = image.var(dim=-1, keepdim=True) ** 0.5
            image = (image - mu) / (sigma + 1e-6)

        loss = (predicted_image - image) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss


if __name__ == "__main__":
    inp = torch.randn(3, 1, 1280, 768)

    model = MaskedAutoencoderHiera(
        input_size=(1280, 768),
        in_chans=1,
        patch_kernel=(4, 4),
        patch_stride=(4, 4),
        mask_ratio=0.75,
        use_sin_cos_pos_embed=True,
    )
    loss = MAELoss()

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            output = model(inp)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    patched, pred, pred_mask = model(inp)

    loss_value = loss(patched, pred, pred_mask)
