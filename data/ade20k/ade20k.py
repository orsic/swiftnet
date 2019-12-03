from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat
import numpy as np


def init_ade20k_class_color_info(path: Path):
    colors = loadmat(str(path / 'color150.mat'))['colors']
    classes = []
    with (path / 'object150_info.csv').open('r') as f:
        for i, line in enumerate(f.readlines()):
            if bool(i):
                classes += [line.rstrip().split(',')[-1]]
    return classes + ['void'], np.concatenate([colors, np.array([[0, 0, 0]], dtype=colors.dtype)])


class_info, color_info = init_ade20k_class_color_info(Path('/home/morsic/datasets/ADE20k'))


class ADE20k(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 150

    def __init__(self, root: Path, transforms: lambda x: x, subset='training', open_images=True, epoch=None):
        self.root = root
        self.open_images = open_images
        self.images_dir = root / 'ADEChallengeData2016/images/' / subset
        self.labels_dir = root / 'ADEChallengeData2016/annotations/' / subset

        self.images = list(sorted(self.images_dir.glob('*.jpg')))
        self.labels = list(sorted(self.labels_dir.glob('*.png')))

        self.transforms = transforms
        self.subset = subset
        self.epoch = epoch

        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'name': self.images[item].stem,
            'subset': self.subset,
            'labels': self.labels[item]
        }
        if self.open_images:
            ret_dict['image'] = self.images[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)
        return self.transforms(ret_dict)
