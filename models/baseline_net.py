import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    def __init__(self, in_channels, out_features):
        super(BaseNet, self).__init__()
        self.cnn_block1 = self.cnn_block(in_channels, 32)
        self.cnn_block2 = self.cnn_block(32, 48)
        self.cnn_block3 = self.cnn_block(48, 64)
        self.cnn_block4 = self.cnn_block(64, 64)
        self.cnn_block5 = self.cnn_block(64, 32)
        self.linear = nn.Linear(2592, out_features)

    def cnn_block(self,in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = self.cnn_block4(x)
        x = self.cnn_block5(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    model = BaseNet(6, 9)
    tens = torch.randn(8, 6, 300, 300) # batch_size: 8, img_channels 6 (3+3), res_x, res_y
    out = model(tens)
    print(out.shape) # [8,9]


