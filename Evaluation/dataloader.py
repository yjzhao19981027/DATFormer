from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root):
        pred_names = os.listdir(pred_root)
        #label_names = os.listdir(label_root)

        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), pred_names))
        self.label_path = list(
            map(lambda x: os.path.join(label_root, x), pred_names))

    def __getitem__(self, item):
        if not os.path.exists(self.image_path[item][:-4] + ".jpg"):
            pred = Image.open(self.image_path[item][:-4] + ".png").convert('L')
        else:
            pred = Image.open(self.image_path[item][:-4] + ".jpg").convert('L')

        if not os.path.exists(self.label_path[item][:-4] + ".jpg"):
            gt = Image.open(self.label_path[item][:-4] + ".png").convert('L')
        else:
            gt = Image.open(self.label_path[item][:-4] + ".jpg").convert('L')
            
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)

class EvalVideoDataset(data.Dataset):
    def __init__(self, pred_root, label_root, test_name):
        self.image_path = []
        self.label_path = []
        pred_root = os.path.join(pred_root, test_name)
        for video_name in os.listdir(pred_root):
            pre_root = os.path.join(pred_root, video_name)
            for pred in os.listdir(pre_root):
                self.image_path.append(os.path.join(pre_root, pred))
                self.label_path.append(os.path.join(label_root, test_name, video_name[:-4], pred))

    def __getitem__(self, item):
        if not os.path.exists(self.image_path[item][:-4] + ".jpg"):
            pred = Image.open(self.image_path[item][:-4] + ".png").convert('L')
        else:
            pred = Image.open(self.image_path[item][:-4] + ".jpg").convert('L')

        if not os.path.exists(self.label_path[item][:-4] + ".jpg"):
            gt = Image.open(self.label_path[item][:-4] + ".png").convert('L')
        else:
            gt = Image.open(self.label_path[item][:-4] + ".jpg").convert('L')
            
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)