import torch
from perceiver_pytorch import Perceiver as PerceiverTorch
import torch.nn as nn


class Perceiver(nn.Module):
    def __init__(self, input_channels, num_outputs):
        super(Perceiver, self).__init__()
        self.perceiver = PerceiverTorch(
                input_channels = input_channels,          # number of channels for each token of the input
                input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
                num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
                max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
                depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                            #   depth * (cross attention -> self_per_cross_attn * self attention)
                num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = 512,            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8
                cross_dim_head = 64,         # number of dimensions per cross attention head
                latent_dim_head = 64,        # number of dimensions per latent self attention head
                num_classes = num_outputs,          # output number of classes
                attn_dropout = 0.,
                ff_dropout = 0.,
                weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
                fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
                self_per_cross_attn = 2      # number of self attention blocks per cross attention

        )

    def forward(self, x):
        x = x.permute((0,2,3,1))
        return self.perceiver(x)



"""
                num_freq_bands=6,
                depth=6,
                max_freq=10, 
                input_channels=input_channels, 
                num_classes=num_outputs
                """


if __name__ == '__main__':
    tens = torch.randn((1,6,320,320))
    model = Perceiver(6, 9)

    out=model(tens)
    print(out.shape)
