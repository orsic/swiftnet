import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from pathlib import Path
import numpy as np
import os

from models.semseg import SemsegModel
from models.resnet.resnet_single_scale import *
from data.transform import *
from data.mux.transform import *
from data.cityscapes import Cityscapes
from models.util import get_n_params
from evaluation import StorePreds, StoreSubmissionPreds


root = Path('datasets/Cityscapes')
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

scale = 255
mean = Cityscapes.mean
std = Cityscapes.std
mean_rgb = tuple(np.uint8(scale * np.array(mean)))

class_info = Cityscapes.class_info
color_info = Cityscapes.color_info

downsample = 1
num_levels = 1
alphas = [1.]
target_size = ts = (2048, 1024)
target_size_feats = (ts[0] // 4, ts[1] // 4)

eval_each = 4

trans_train = trans_val = Compose(
    [Open(),
     RemapLabels(Cityscapes.map_to_id, Cityscapes.num_classes),
     Pyramid(alphas=alphas),
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     Normalize(scale, mean, std),
     Tensor(),
     ]
)

dataset_train = Cityscapes(root, transforms=trans_train, subset='train')
dataset_val = Cityscapes(root, transforms=trans_val, subset='val')

use_bn = True 
resnet = resnet18(pretrained=False, efficient=False, use_bn=use_bn)
model = SemsegModel(resnet, Cityscapes.num_classes, use_bn=use_bn)
model.load_state_dict(torch.load('weights/swiftnet_ss_cs.pt'), strict=True)

batch_size = 1 

nw = 8 
loader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=custom_collate, num_workers=nw)
loader_val = DataLoader(dataset_val, batch_size=batch_size, collate_fn=custom_collate, num_workers=nw)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
spp_params = get_n_params(model.backbone.spp.parameters())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')
print(f'SPP params: {spp_params:,}')

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
    eval_observers += [sp]
