import torch.nn as nn

class TactileVisionEncoder(nn.Module):
    def __init__(self, output_size=(32, 3, 2)):
        super(TactileVisionEncoder, self).__init__()
        self.num_channels = output_size[0]
        self.conv1 = nn.Conv2d(1, self.num_channels//2, kernel_size=3, stride=1, padding=1)  # Output: (16, H, W)
        self.conv2 = nn.Conv2d(self.num_channels//2, self.num_channels, kernel_size=3, stride=1, padding=1)  # Output: (32, H, W)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size[1:])  # Output: (32, 1, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))       # Output: (N, 16, H, W)
        x = self.relu(self.conv2(x))       # Output: (N, 32, H, W)
        x = self.global_pool(x)            # Output: (N, 32, 1, 1)
        return x
