import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as FT
from settings import path_to_images, path_to_annotation, path_to_empty_images
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    img_files = [x for x in os.listdir(dir) if is_image_file(x)]
    return img_files


class Data(Dataset):
    def __init__(self, path_to_images=path_to_images,
                 path_to_annotation=path_to_annotation,
                 path_to_empty_images=path_to_empty_images):
        super(Data, self).__init__()
        self.path_to_images = path_to_images
        self.path_to_annotation = path_to_annotation
        self.path_to_empty_images = path_to_empty_images
        self.obj_img_files = make_dataset(path_to_images)
        self.blank_img_files = make_dataset(path_to_empty_images)
        self.dataset_len = min(len(self.obj_img_files), len(self.blank_img_files))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        # open #item object image
        path_to_obj_img = os.path.join(self.path_to_images, self.obj_img_files[item])
        obj_image = Image.open(path_to_obj_img, 'RGB')
        obj_image = self.image_to_tensor(obj_image)

        # read corresponding annotation file
        objects = []
        path_to_ann_file = os.path.join(self.path_to_annotation,
                                        ''.join(self.obj_img_files[item].split('.')[:-1])+'.txt')
        with open(path_to_ann_file) as ann_file:
            for line in ann_file:
                objects.append([float(x) for x in line.strip().split(' ')])
        objects = torch.tensor(objects, dtype=torch.float32)
        obj_window = self.get_window(*objects[1:])

        # open #item blank image
        path_to_empty_image = os.path.join(self.path_to_empty_images, self.obj_img_files[item])
        empty_image = Image.open(path_to_empty_image, 'RGB')
        empty_image = self.image_to_tensor(empty_image)

        # read random annotation file
        random_index = np.random.randint(0, self.dataset_len, 1, dtype=np.int32)
        path_to_random_ann_file = os.path.join(self.path_to_annotation,
                                               ''.join(self.obj_img_files[random_index].split('.')[:-1]) + '.txt')
        with open(path_to_random_ann_file) as random_ann_file:
            cls, cx, cy, w, h = next(random_ann_file)
        empty_window = self.get_window(cx, cy, w, h)

        return {
            'obj_img': obj_image,
            'objects': objects,
            'obj_w': obj_window,
            'empty_img': empty_image,
            'empty_w': empty_window
        }

    def image_to_tensor(self, img):
        tensor = FT.to_tensor(img)
        tensor = FT.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor

    def get_window(self, cx, cy, w, h):
        window = torch.zeros((1, 256, 256), dtype=torch.float32)
        window[:, torch.round((cy-h/2)*256).int():torch.round((cy+h/2)*256).int(),
                  torch.round((cx-w/2)*256).int():torch.round((cx+w/2)*256).int()] = 1.
        return window

    def collate_fn(self, batch):

        obj_images = []
        obj_windows = []
        obj_objects = []
        empty_images = []
        empty_windows = []

        for d in batch:
            obj_images.append(d)
            obj_windows.append(d)
            obj_objects.append(d)
            empty_images.append(d)
            empty_windows.append(d)

        return {
            'obj_imgs': torch.stack(obj_images, dim=0),
            'obj_w': torch.stack(obj_windows, dim=0),
            'obj_objs': obj_objects,
            'empty_imgs': torch.stack(empty_images, dim=0),
            'empty_w': torch.stack(empty_windows, dim=0)
        }


def get_dataloader(batch_size, num_workers):
    from torch.utils.data import DataLoader
    return DataLoader(Data, batch_size=batch_size, num_workers=num_workers)