from torch.utils.data import Dataset
from pathlib import Path

from .labels import labels

class_info = [label.name for label in labels if label.ignoreInEval is False]
color_info = [label.color for label in labels if label.ignoreInEval is False]

color_info += [[0, 0, 0]]

map_to_id = {}
inst_map_to_id = {}
i, j = 0, 0
for label in labels:
    if label.ignoreInEval is False:
        map_to_id[label.id] = i
        i += 1
        if label.hasInstances is True:
            inst_map_to_id[label.id] = j
            j += 1

id_to_map = {id: i for i, id in map_to_id.items()}
inst_id_to_map = {id: i for i, id in inst_map_to_id.items()}


class Cityscapes(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 19

    map_to_id = map_to_id
    id_to_map = id_to_map

    inst_map_to_id = inst_map_to_id
    inst_id_to_map = inst_id_to_map

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root: Path, transforms: lambda x: x, subset='train', open_depth=False, labels_dir='labels', epoch=None):
        self.root = root
        self.images_dir = self.root / 'rgb' / subset
        self.labels_dir = self.root / labels_dir / subset
        self.depth_dir = self.root / 'depth' / subset
        self.subset = subset
        self.has_labels = subset != 'test'
        self.open_depth = open_depth
        self.images = list(sorted(self.images_dir.glob('*/*.ppm')))
        if self.has_labels:
            self.labels = list(sorted(self.labels_dir.glob('*/*.png')))
        self.transforms = transforms
        self.epoch = epoch

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
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)
        return self.transforms(ret_dict)
