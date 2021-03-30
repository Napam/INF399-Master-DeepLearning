import utils
seed = 42069
utils.seed_everything(seed)

from typing import Generic, Optional, Tuple, List, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils import debugt, debugs, debug
from datetime import datetime

import fishdetr3d as detr
# import detr_batchboy_regular as detr
from generators import Torch3DDataset
from tqdm import tqdm

import sys
# sys.path.append('./detr_custom/')
# from models.matcher import HungarianMatcher
# from models.detr import SetCriterion
from hungarianmatcher import HungarianMatcher
from setcriterion import SetCriterion
import os
import sqlite3
from train_fishdetr3d import _validate_model


if __name__ == '__main__':
    try:
        device = utils.pytorch_init_janus_gpu(1)
        print(f'Using device: {device} ({torch.cuda.get_device_name()})')
        print(utils.get_cuda_status(device))
    except AssertionError as e:
        print('GPU could not initialize, got error:', e)
        device = torch.device('cpu')
        print('Device is set to CPU')

    TORCH_CACHE_DIR = 'torch_cache'
    DATASET_DIR = '/mnt/blendervol/3d_data_test'
    num2name = eval(open(os.path.join(DATASET_DIR,"metadata.txt"), 'r').read())
    TABLE = "bboxes_full"
    WEIGHTS_DIR = 'fish_statedicts_3d'
    torch.hub.set_dir(TORCH_CACHE_DIR)

    model = detr.FishDETR().to(device)
    model.load_state_dict(torch.load('fish_statedicts_3d/weights_2021-03-25/detr_statedicts_epoch9_train0.1433_val0.1391_2021-03-25T03:58:51.pth')['model_state_dict'])
    # model.load_state_dict(torch.load('last_epoch_detr_3d.pth'))

    datagen_range = (0,8)
    datagen = Torch3DDataset(DATASET_DIR, TABLE, 1, shuffle=False, imgnrs=range(*datagen_range))
    BATCH_SIZE = 8
    dataloader = DataLoader(
        dataset = datagen,
        batch_size = BATCH_SIZE,
        collate_fn = detr.collate,
    )

    weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1, 'loss_smooth':1}
    losses = ['labels', 'boxes_3d']
    matcher = HungarianMatcher(use_giou=False, smooth_l1=False)
    criterion = SetCriterion(6, matcher, weight_dict, eos_coef = 0.5, losses=losses)
    criterion = criterion.to(device)

    context = {
        'valloader':dataloader,
        'model':model,
        'criterion':criterion,
        'device':device,
    }

    _, output, loss, running_loss = _validate_model(context, {})
    debug(loss)
    
    df = detr.postprocess_to_df(datagen.indices, output, 0.7).to_csv('nogit_test_output.csv')
    pd.read_sql(f"""
    SELECT * FROM bboxes_full 
    WHERE imgnr IN ({','.join(map(str, datagen.indices))})
    """, datagen.con).to_csv("nogit_test_labels.csv", index=False)