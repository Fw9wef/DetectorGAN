import torch
from torch.utils.data import Dataset
from settings import path_to_images, path_to_annotation, path_to_empty_images
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
        path_to_real_img = os.path.join(self.path_to_images)
        # read corresponding annotation file
        # open #item blank image
        # read random annotation file
        return item

    def collate_fn(self, batch):

        real_images = []
        real_windows = []
        real_objects = []
        empty_images = []
        empty_windows = []

        for d in batch:
            real_images.append(d)
            real_windows.append(d)
            real_objects.append(d)
            empty_images.append(d)
            empty_windows.append(d)

        return {
            'real_imgs': torch.stack(real_images, dim=0),
            'real_w': torch.stack(real_windows, dim=0),
            'real_objs': real_objects,
            'empty_imgs': torch.stack(empty_images, dim=0),
            'empty_w': torch.stack(empty_windows, dim=0)
        }


def get_dataloader(batch_size, num_workers):
    from torch.utils.data import DataLoader
    return DataLoader(Data, batch_size=batch_size, num_workers=num_workers)