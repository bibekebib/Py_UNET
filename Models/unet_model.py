import torch
import torch.nn as nn
from unet_layer import ThreeConvBlock, DownStage, UpStage

class UNET(nn.Module):
    def __init__(self, n_channels, n_classes=1):
        super(UNET, self).__init__()
        # self.nclasses = n_classes
        # self.n_channels = n_channels

        # Define the initial block (ThreeConvBlock) of the UNET
        self.start = ThreeConvBlock(n_channels, 64)

        # Define the down-sampling stages (DownStage) of the UNET
        self.down1 = DownStage(64, 128)
        self.down2 = DownStage(128, 256)
        self.down3 = DownStage(256, 512)
        self.down4 = DownStage(512, 1024)

        # Define the up-sampling stages (UpStage) of the UNET
        self.up1 = UpStage(1024, 512)
        self.up2 = UpStage(512, 256)
        self.up3 = UpStage(256, 128)
        self.up4 = UpStage(128, 64)
        print(n_channels)
        # Output layer (final convolution) for producing segmentation masks
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)  # Update the number of output channels
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the UNET architecture

        # Initial block
        x_start = self.start(x)
        # print("x_start shape:", x_start.shape)

        # Down-sampling stages
        x_down1 = self.down1(x_start)
        # print("x_down1 shape:", x_down1.shape)
        x_down2 = self.down2(x_down1)
        # print("x_down2 shape:", x_down2.shape)
        x_down3 = self.down3(x_down2)
        # print("x_down3 shape:", x_down3.shape)
        x_down4 = self.down4(x_down3)
        # print("x_down4 shape:", x_down4.shape)

        # Up-sampling stages with skip connections
        x_up1 = self.up1(x_down4, x_down3)
        # print("x_up1 shape:", x_up1.shape)
        x_up2 = self.up2(x_up1, x_down2)
        # print("x_up2 shape:", x_up2.shape)
        x_up3 = self.up3(x_up2, x_down1)
        # print("x_up3 shape:", x_up3.shape)
        x_up4 = self.up4(x_up3, x_start)
        # print("x_up4 shape:", x_up4.shape)

        # Final output layer (convolution) for producing segmentation masks
        x_out = self.out(x_up4)
        # print("x_out shape:", x_out.shape)

        # Final output layer (convolution) for producing segmentation masks
        x_out = self.out(x_up4)
        segmentation_mask = self.sigmoid(x_out)
        # print(segmentation_mask)
        return segmentation_mask
