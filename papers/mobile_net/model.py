import torch
from torch import nn
from torch.nn import functional as nn_func


"""
Implementation of MobileNets:
"""



class DepthSeptConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block.

    This block performs depthwise separable convolutions, which consist of:
    1. A depthwise convolution: Applies a separate 3x3 convolutional filter for each input channel.
    2. A pointwise convolution: Applies a 1x1 convolution to mix the features from the depthwise convolution.

    The block includes batch normalization and ReLU activation applied after each convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride of the depthwise convolution. Default is 1.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Depthwise convolution followed by batch normalization and ReLU activation.
        self.depthwise_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=3,
                      stride=stride,
                      groups=in_channels,
                      padding=1),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True)
        )

        # Pointwise convolution followed by batch normalization and ReLU activation.
        self.pointwise_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Depthwise Separable Convolution Block.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height, width).
        """
        x = self.depthwise_layer(x)
        x = self.pointwise_layer(x)

        return x

print("The Depthwise Separable Convolutions block defined!")



class MobileNet(nn.Module):
    """
    MobileNet Architecture from the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"


    MobileNet is a lightweight deep learning model designed for efficient inference on mobile and edge devices.
    It uses depthwise separable convolutions to reduce the number of parameters and computational complexity.

    Args:
        out_channels (int, optional): Number of output channels for the initial convolution layer. Default is 32.
        output_size (int, optional): Size of the output features before the fully connected layer. Default is 1024.
        num_classes (int, optional): Number of classes for classification. Default is 10.
    """

    def __init__(self, out_channels=32, output_size=1024, num_classes=10):
        super().__init__()

        # Initial convolutional layer followed by batch normalization and ReLU activation.
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      kernel_size=3,
                      out_channels=out_channels,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

        # MobileNet core: sequence of depthwise separable convolution blocks.
        self.mobile_net = nn.Sequential(
            DepthSeptConvBlock(in_channels=32, out_channels=64, stride=1),
            DepthSeptConvBlock(in_channels=64, out_channels=128, stride=2),
            DepthSeptConvBlock(in_channels=128, out_channels=128, stride=1),
            DepthSeptConvBlock(in_channels=128, out_channels=256, stride=2),
            DepthSeptConvBlock(in_channels=256, out_channels=256, stride=1),
            DepthSeptConvBlock(in_channels=256, out_channels=512, stride=2),

            # Repeated (5x 512) Depthwise Separable Convolution Blocks.
            DepthSeptConvBlock(in_channels=512, out_channels=512, stride=1),
            DepthSeptConvBlock(in_channels=512, out_channels=512, stride=1),
            DepthSeptConvBlock(in_channels=512, out_channels=512, stride=1),
            DepthSeptConvBlock(in_channels=512, out_channels=512, stride=1),
            DepthSeptConvBlock(in_channels=512, out_channels=512, stride=1),

            # Additional Depthwise Separable Convolutions followed by Average Pooling.
            DepthSeptConvBlock(in_channels=512, out_channels=1024, stride=2),
            DepthSeptConvBlock(in_channels=1024, out_channels=1024, stride=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully connected layer for classification.
        self.fc = nn.Sequential(
            nn.Linear(in_features=output_size, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MobileNet network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, num_classes), representing the class scores.
        """
        x = self.input_conv(x)
        x = self.mobile_net(x)

        # Flatten the output before the fully connected layer.
        x = x.view(-1, 1024)
        x = self.fc(x)
        x = nn_func.softmax(x, dim=1)

        return x

print("MobileNet architecture with depthwise separable convolutions defined!")
