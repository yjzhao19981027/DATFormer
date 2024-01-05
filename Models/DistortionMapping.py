import torch.nn as nn
import torch
import torchvision


class DistortionMapping(nn.Module):
    def __init__(self, input_channel=128, output_channel=128):
        super(DistortionMapping, self).__init__()

        self.kernel_size = kernel_size = 3
        self.stride = stride = 1
        self.padding = padding = 1
        self.proj = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        # --- deformable offset and modulator
        self.offset_conv = nn.Conv2d(input_channel, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(input_channel, 1 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.norm = nn.BatchNorm2d(output_channel)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def deform_proj(self, x):
        att_map = x

        max_offset = min(att_map.shape[-2], att_map.shape[-1]) // 4
        offset = self.offset_conv(att_map).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(att_map))
        att_map = torchvision.ops.deform_conv2d(input=att_map,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        att_map = self.act(self.norm(att_map))
        return att_map

    def forward(self, x):
        fea = x
        att_map = self.deform_proj(fea)
        
        return x * 2 * self.sigmoid(att_map)