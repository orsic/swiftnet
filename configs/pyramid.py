import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from pathlib import Path
import os
import numpy as np

from models.semseg import SemsegPyramidModel
from models.resnet.resnet_pyramid import *
from data.transform import *
from data.mux.transform import *
from data.cityscapes import Cityscapes
from evaluation import StorePreds, StoreSubmissionPreds

from models.util import get_n_params

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
root = Path('datasets/Cityscapes')

scale = 255
mean = Cityscapes.mean
std = Cityscapes.std
mean_rgb = tuple(np.uint8(scale * np.array(mean)))

class_info = Cityscapes.class_info
color_info = Cityscapes.color_info

downsample = 1
alpha = 2.0
num_levels = 3
alphas = [alpha ** i for i in range(num_levels)]
target_size = (2048, 1024)
target_size_feats = (2048 // 4, 1024 // 4)

eval_each = 4

trans_train = trans_val = Compose(
    [Open(),
     RemapLabels(Cityscapes.map_to_id, Cityscapes.num_classes),
     Pyramid(alphas=alphas),
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     Normalize(255, mean, std),
     Tensor(),
     ]
)

dataset_train = Cityscapes(root, transforms=trans_train, subset='train')
dataset_val = Cityscapes(root, transforms=trans_val, subset='val')

use_bn = True 
resnet = resnet18(pretrained=True, pyramid_levels=num_levels, efficient=False, use_bn=use_bn)
model = SemsegPyramidModel(resnet, Cityscapes.num_classes)
model.load_state_dict(torch.load('weights/swiftnet_pyr_cs.pt'), strict=True)

batch_size = bs = 1

loader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate, num_workers=8)
loader_train = DataLoader(dataset_train, batch_size=bs, collate_fn=custom_collate)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')

eval_loaders = [(loader_val, 'val')]
eval_observers = []
store_preds = False
if store_preds:
    store_dir = f'{dir_path}/out/'
    store_dir_color = f'{dir_path}/outc/'
    for d in ['', 'val']:
        os.makedirs(store_dir + d, exist_ok=True)
        os.makedirs(store_dir_color + d, exist_ok=True)
    to_color = ColorizeLabels(Cityscapes.color_info)
    to_image = Compose([DenormalizeTh(scale, mean, std), Numpy(), to_color])
    sp = StoreSubmissionPreds(store_dir, lambda x: x, to_color, store_dir_color)
