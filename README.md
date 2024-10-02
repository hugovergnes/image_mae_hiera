# Image MAE for Hiera
Hierarchical ViT applied to MAE image pre training

In this project, I'm looking at applying Image MAE [1] to Hiera [2] while using much less resources.

If you want to learn more or have something to discuss please feel free to contact me or checkout the sources for the original research!

# Context

Image pre-training has been at the center of research for a lot of state of the art models, but the computational requirements are prohibitive. In the original Video MAE paper [3], they use 128 A100 GPUs! Very few groups can do that. 


# References

[1] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16000-16009).

[2] Ryali, C., Hu, Y. T., Bolya, D., Wei, C., Fan, H., Huang, P. Y., ... & Feichtenhofer, C. (2023, July). Hiera: A hierarchical vision transformer without the bells-and-whistles. In International Conference on Machine Learning (pp. 29441-29454). PMLR.

[3] Tong, Zhan, et al. "Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training." Advances in neural information processing systems 35 (2022): 10078-10093.