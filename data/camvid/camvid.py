from torch.utils.data import Dataset
from pathlib import Path

class_info = ['building', 'tree', 'sky', 'car', 'sign', 'road', 'pedestrian', 'fence', 'column pole', 'sidewalk',
              'bicyclist']
color_info = [(128, 0, 0), (128, 128, 0), (128, 128, 128), (64, 0, 128), (192, 128, 128), (128, 64, 128), (64, 64, 0),
              (64, 74, 128), (192, 192, 128), (0, 0, 192), (0, 128, 192)]

color_info += [[0, 0, 0]]


class CamVid(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = len(class_info)

    mean = [111.376, 63.110, 83.670]
    std = [41.608, 54.237, 68.889]

    def __init__(self, root: Path, transforms: lambda x: x, subset='train'):
        self.root = root
        self.subset = subset
        self.image_names = [line.rstrip() for line in (root / f'{subset}.txt').open('r').readlines()]
        name_filter = lambda x: x.name in self.image_names
        self.images = list(filter(name_filter, (self.root / 'rgb').iterdir()))
        self.labels = list(filter(name_filter, (self.root / 'labels/ids').iterdir()))
        self.transforms = transforms
        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'image': self.images[item],
            'name': self.images[item].stem,
            'subset': self.subset,
            'labels': self.labels[item]
        }
        return self.transforms(ret_dict)
