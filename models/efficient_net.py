from efficientnet_pytorch import EfficientNet #https://github.com/lukemelas/EfficientNet-PyTorch
import torch
import torch.nn as nn


class EffNetB3(nn.Module):
    def __init__(self, in_channels, out_features=9):
        super(EffNetB3, self).__init__()
        drop_connect = 0.0
        dropout = 0.0
        self.model = EfficientNet.from_name('efficientnet-b3', in_channels=in_channels, num_classes=out_features, dropout_rate=dropout, drop_connect_rate=drop_connect)
        self.model._fc = nn.Linear(1536, out_features)
        self.model._dropout_training=False

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = EffNetB3(6,9)
    tens = torch.randn((8,6,300,300))
    out = model(tens)
    print(out.shape)






