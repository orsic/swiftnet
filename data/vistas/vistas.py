from torch.utils.data import Dataset
from pathlib import Path

class_info = ['animal--bird', 'animal--ground-animal', 'construction--barrier--curb', 'construction--barrier--fence',
              'construction--barrier--guard-rail', 'construction--barrier--other-barrier',
              'construction--barrier--wall', 'construction--flat--bike-lane', 'construction--flat--crosswalk-plain',
              'construction--flat--curb-cut', 'construction--flat--parking', 'construction--flat--pedestrian-area',
              'construction--flat--rail-track', 'construction--flat--road', 'construction--flat--service-lane',
              'construction--flat--sidewalk', 'construction--structure--bridge', 'construction--structure--building',
              'construction--structure--tunnel', 'human--person', 'human--rider--bicyclist',
              'human--rider--motorcyclist', 'human--rider--other-rider', 'marking--crosswalk-zebra', 'marking--general',
              'nature--mountain', 'nature--sand', 'nature--sky', 'nature--snow', 'nature--terrain',
              'nature--vegetation', 'nature--water', 'object--banner', 'object--bench', 'object--bike-rack',
              'object--billboard', 'object--catch-basin', 'object--cctv-camera', 'object--fire-hydrant',
              'object--junction-box', 'object--mailbox', 'object--manhole', 'object--phone-booth', 'object--pothole',
              'object--street-light', 'object--support--pole', 'object--support--traffic-sign-frame',
              'object--support--utility-pole', 'object--traffic-light', 'object--traffic-sign--back',
              'object--traffic-sign--front', 'object--trash-can', 'object--vehicle--bicycle', 'object--vehicle--boat',
              'object--vehicle--bus', 'object--vehicle--car', 'object--vehicle--caravan', 'object--vehicle--motorcycle',
              'object--vehicle--on-rails', 'object--vehicle--other-vehicle', 'object--vehicle--trailer',
              'object--vehicle--truck', 'object--vehicle--wheeled-slow', 'void--car-mount', 'void--ego-vehicle',
              'void--unlabeled']
color_info = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180], [102, 102, 156],
              [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
              [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232], [150, 100, 100], [70, 70, 70],
              [150, 120, 90], [220, 20, 60], [255, 0, 0], [255, 0, 0], [255, 0, 0], [200, 128, 128], [255, 255, 255],
              [64, 170, 64], [128, 64, 64], [70, 130, 180], [255, 255, 255], [152, 251, 152], [107, 142, 35],
              [0, 170, 30], [255, 255, 128], [250, 0, 30], [0, 0, 0], [220, 220, 220], [170, 170, 170], [222, 40, 40],
              [100, 170, 30], [40, 40, 40], [33, 33, 33], [170, 170, 170], [0, 0, 142], [170, 170, 170],
              [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 142], [250, 170, 30], [192, 192, 192],
              [220, 220, 0], [180, 165, 180], [119, 11, 32], [0, 0, 142], [0, 60, 100], [0, 0, 142], [0, 0, 90],
              [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32], [0, 0, 0],
              [0, 0, 0]]


class Vistas(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 63

    def __init__(self, root: Path, transforms: lambda x: x, subset='training', open_images=True, epoch=None):
        self.root = root
        self.open_images = open_images
        self.images_dir = root / subset / 'images'
        self.labels_dir = root / subset / 'labels'

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
