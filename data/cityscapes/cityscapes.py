from torch.utils.data import Dataset
from pathlib import Path

from .labels import labels

class_info = [label.name for label in labels if label.ignoreInEval is False]
color_info = [label.color for label in labels if label.ignoreInEval is False]

color_info += [[0, 0, 0]]

map_to_id = {}
i = 0
for label in labels:
    if label.ignoreInEval is False:
        map_to_id[label.id] = i
        i += 1

id_to_map = {id: i for i, id in map_to_id.items()}


class Cityscapes(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 19

    map_to_id = map_to_id
    id_to_map = id_to_map

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root: Path, transforms: lambda x: x, subset='train'):
        self.root = root
        self.images_dir = self.root / 'img/left/leftImg8bit' / subset
        self.labels_dir = self.root / 'gtFine' / subset
        self.subset = subset
        self.has_labels = subset != 'test'
        self.images = list(sorted(self.images_dir.glob('*/*.png')))
        if self.has_labels:
            self.labels = list(sorted(self.labels_dir.glob('*/*_gtFine_labelIds.png')))
        self.transforms = transforms
        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'image': self.images[item],
            'name': self.images[item].stem,
            'subset': self.subset,
        }
        if self.has_labels:
            ret_dict['labels'] = self.labels[item]
        return self.transforms(ret_dict)
