import torch
import torch.nn as nn
import math

class myWeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(myWeightedBCEWithLogitsLoss, self).__init__()

    def myBCEWithLogitsLoss(self, output, target, matrix, weight=None, pos_weight=None):
        output = torch.sigmoid(output)
        # 处理正负样本不均衡问题
        if pos_weight is None:
            label_size = output.size()[1]
            pos_weight = torch.ones(label_size)
        # 处理多标签不平衡问题
        if weight is None:
            label_size = output.size()[1]
            weight = torch.ones(label_size)

        val = 0
        for li_x, li_y in zip(output, target):
            for i, xy in enumerate(zip(li_x, li_y)):
                x, y = xy
                eps = 1e-07
                loss_val = pos_weight[i] * y * torch.log(x + eps) + (1 - y) * torch.log(1 - x + eps)
                val += weight[i] * loss_val/ x.shape[0] / x.shape[1]
        tot = torch.sum(val)
        return -tot / (output.size()[0] * output.size(1))

    def forward(self, image, label):
        _, _, h, w = image.shape
        matrix = torch.ones((h, w))
        for i in range(h):
            half_h = h / 2
            matrix[i, :] = math.cos(abs(i - half_h) / half_h * math.pi / 2) * 2
        matrix = matrix.cuda()
        loss = self.myBCEWithLogitsLoss(image, label, matrix)
        return loss