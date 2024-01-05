import torch
import torch.nn as nn
import math
import torchvision
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch import Tensor

class DWConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1, bias=False):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,
                                   padding=padding, groups=in_chans, bias=bias)
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)

        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DeformableCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(DeformableCNN, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.offset_modulator_conv = DWConv2d(in_channels, 3 * in_channels)

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.reset_parameters()


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        B, C, H, W = input.size()
        offset_modulator = self.offset_modulator_conv(input)
        offset_y, offset_x, modulator = torch.chunk(offset_modulator, 3, dim=1)
        modulator = 2. * torch.sigmoid(modulator)
        offset = torch.cat((offset_y, offset_x), dim=1)
        max_offset = max(H, W) // 4
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(input=input,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation
                                          )
        x = self.act(self.norm(x))
        return x


class DConvMLP(nn.Module):
    def __init__(self, in_chans=64, emb_chans=None, out_chans=None, act_layer=nn.GELU, drop=0., time_chans=1):
        super().__init__()
        # spatial deformable proj

        out_chans = out_chans or in_chans
        emb_chans = emb_chans or in_chans
        self.T = time_chans
        self.adapter_dcnn = DeformableCNN(in_chans, emb_chans, (3, 3), 1, 0)
        self.c_mlp = nn.Linear(emb_chans, out_chans)
        self.act = act_layer()
        self.proj = nn.Linear(out_chans, out_chans)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, THW, C = x.shape
        HW = THW // self.T
        H = int(math.sqrt(HW // 2))
        W = int(H * 2)
        x = x.reshape(B * self.T, H, W, C).permute(0, 3, 1, 2)

        wh = self.adapter_dcnn(x).permute(0, 2, 3, 1).reshape(B, self.T * HW, -1)
        x = x.permute(0, 2, 3, 1).reshape(B, self.T * HW, -1)
        c = self.c_mlp(wh)
        x = self.act(x + c)
        x = self.proj(x)
        x = self.drop(x)
        return x