from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        padding_mode="zeros",
        norm_layer=nn.InstanceNorm2d,
        bias=True,
        scale_factor=1,
    ):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * scale_factor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                groups=in_channels,
                bias=bias,
            ),
            norm_layer(in_channels * scale_factor),
            nn.Conv2d(
                in_channels=in_channels * scale_factor,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.conv(x)
