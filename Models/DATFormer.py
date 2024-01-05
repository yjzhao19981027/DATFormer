import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_extractor import feature_extractor

class DATFormer(nn.Module):
    def __init__(self, args):
        super(DATFormer, self).__init__()

        # Encoder
        token_dim = 64
        self.fea_extractor = feature_extractor(args=args)

        self.sal_fea_extractor = nn.Conv2d(token_dim, 1, kernel_size=(1, 1))
        self.con_fea_extractor = nn.Conv2d(token_dim, 1, kernel_size=(1, 1))

    def forward(self, image_Input):
        # B, 64, 224, 448
        B = image_Input.shape[0]
        image_fea1_1, image_fea1_4, image_fea1_8  = self.fea_extractor(image_Input)
        
        fea1_1 = image_fea1_1
        fea1_4 = image_fea1_4
        fea1_4 = F.interpolate(fea1_4, size=(224, 448), mode='bilinear', align_corners=True)
        fea1_8 = image_fea1_8
        fea1_8 = F.interpolate(fea1_8, size=(224, 448), mode='bilinear', align_corners=True)

        weights = [1, 0.8, 0.5]

        fea = fea1_1 * weights[0] + fea1_4 * weights[1] + fea1_8 * weights[2]

        sal_fea = self.sal_fea_extractor(fea)
        con_fea = self.con_fea_extractor(fea)

        return sal_fea, con_fea
