from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random
import os


def load_list(dataset_name, data_root):
    images = []
    labels = []
    contours = []
    
    img_root = os.path.join(data_root, dataset_name, "train/images")
    label_root = os.path.join(data_root, dataset_name, "train/labels")
    contour_root = os.path.join(data_root, dataset_name, "train/contours")

    for img in os.listdir(img_root):
        images.append(os.path.join(img_root, img))
        if os.path.exists(os.path.join(label_root, img[:-4] + ".jpg")):
            labels.append(os.path.join(label_root, img[:-4] + ".jpg"))
        else:
            labels.append(os.path.join(label_root, img[:-4] + ".png"))
        if os.path.exists(os.path.join(contour_root, img[:-4] + ".jpg")):
            contours.append(os.path.join(contour_root, img[:-4] + ".jpg"))
        else:
            contours.append(os.path.join(contour_root, img[:-4]+".png"))

    return images, labels, contours


def load_test_list(dataset_name, data_root):
    images = []

    img_root = os.path.join(data_root, dataset_name, "test/images")

    for img in os.listdir(img_root):
        images.append(os.path.join(img_root, img))

    return images


class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, mode, img_size=224, scale_size=None, t_transform=None,
                 method=None):
        self.dataset_name = dataset_list

        if mode == 'train':
            self.image_path, self.label_path, self.contour_path = load_list(dataset_list, data_root)
        else:
            self.image_path = load_test_list(dataset_list, data_root)

        self.transform = transform
        self.t_transform = t_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size
        self.method = method

        self.height_size = img_size
        self.width_size = img_size * 2

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]

        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])

        if self.mode == 'train':

            label = Image.open(self.label_path[item]).convert('L')
            contour = Image.open(self.contour_path[item]).convert('L')
            random_size = self.scale_size

            new_img = trans.Scale((random_size * 2, random_size * 1))(image)
            new_label = trans.Scale((random_size * 2, random_size * 1), interpolation=Image.NEAREST)(label)
            new_contour = trans.Scale((random_size * 2, random_size * 1), interpolation=Image.NEAREST)(contour)

            # random crop
            w, h = new_img.size
            if w != self.img_size * 2 and h != self.img_size:
                x1 = random.randint(0, w - self.img_size * 2)
                y1 = random.randint(0, h - self.img_size * 1)
                new_img = new_img.crop((x1, y1, x1 + self.img_size * 2, y1 + self.img_size * 1))
                new_label = new_label.crop((x1, y1, x1 + self.img_size * 2, y1 + self.img_size * 1))
                new_contour = new_contour.crop((x1, y1, x1 + self.img_size * 2, y1 + self.img_size * 1))

            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)
                new_contour = new_contour.transpose(Image.FLIP_LEFT_RIGHT)

            # new_img = trans.Scale((self.width_size, self.height_size))(new_img)
            # new_label = trans.Scale((self.width_size, self.height_size), interpolation=Image.NEAREST)(new_label)
            # new_contour = trans.Scale((self.width_size, self.height_size), interpolation=Image.NEAREST)(new_contour)

            new_img = self.transform(new_img)
            label_224 = self.t_transform(new_label)
            contour_224 = self.t_transform(new_contour)
        
            return new_img, label_224, contour_224,
        else:
            
            new_img  = self.transform(image)
            return new_img, image_w, image_h, self.image_path[item]

    def __len__(self):
        return len(self.image_path)


def get_loader(dataset_list, data_root, img_size, mode='train', method=None, padding=None):
    label_size = img_size
    height_size = img_size
    width_size = img_size * 2
    if mode == 'train':
        transform = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        t_transform = trans.Compose([
            trans.Scale((width_size, height_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
    else:
        transform = trans.Compose([
            trans.Scale((width_size, height_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
        ])
    
    scale_size = 256

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform, mode, img_size, scale_size, t_transform, method=method)
    else:
        dataset = ImageData(dataset_list, data_root, transform, mode, img_size, method=method)

    return dataset
