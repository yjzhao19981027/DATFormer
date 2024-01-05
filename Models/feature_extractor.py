from crypt import methods
import torch
import torch.nn as nn
from .t2t_vit_v2 import T2t_vit_t_14_v2
from .reverse_t2t import Reverse_t2t
from .DensityMap import DensityMap

class feature_extractor(nn.Module):
    def __init__(self, args):
        super(feature_extractor, self).__init__()

        self.density_map = DensityMap()

        self.t2t_backbone = T2t_vit_t_14_v2(pretrained=True, args=args)

        self.reverse_t2t = Reverse_t2t(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)
        
    def forward(self, image_Input):
        density_map = self.density_map(image_Input)

        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.t2t_backbone(image_Input, density_map)

        fea1_1, fea_1_4, fea_1_8 = self.reverse_t2t(rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4)  

        return fea1_1, fea_1_4, fea_1_8
