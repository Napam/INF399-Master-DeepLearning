from fishdetr3d import FishDETR as FishDETR_first
from fishdetr3d_alt import FishDETR as FishDETR_alt
from fishdetr3d_splitfc import FishDETR as FishDETR_split
from fishdetr3d import collate

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
import pathlib

from train_fishdetr3d import train_model 


if __name__ == '__main__':
    try:
        device = utils.pytorch_init_janus_gpu(1)
        print(f'Using device: {device} ({torch.cuda.get_device_name()})')
        print(utils.get_cuda_status(device))
    except (AssertionError, RuntimeError) as e:
        print('GPU could not initialize, got error:', e)
        device = torch.device('cpu')
        print('Device is set to CPU')

    TORCH_CACHE_DIR = 'torch_cache'
    DATASET_DIR = '/mnt/blendervol/3d_data'
    TABLE = 'bboxes_full'
    WEIGHTS_DIR = 'fish_statedicts_3d'
    torch.hub.set_dir(TORCH_CACHE_DIR)
    num2name = eval(open(os.path.join(DATASET_DIR,"metadata.txt"), 'r').read())

    db_con = sqlite3.connect(f'file:{os.path.join(DATASET_DIR,"bboxes.db")}?mode=ro', uri=True)
    print("Getting number of images in database: ", end="")
    n_data = pd.read_sql_query(f'SELECT COUNT(DISTINCT(imgnr)) FROM {TABLE}', db_con).values[0][0]
    print(n_data)

    TRAIN_RANGE = (0, 10000)
    VAL_RANGE = (49000,50000)
    
    TRAIN_RANGE = (0, 8)
    VAL_RANGE = (49000,49000+8)

    print(f"TRAIN_RANGE: {TRAIN_RANGE}")
    print(f"VAL_RANGE: {VAL_RANGE}")

    traingen = Torch3DDataset(DATASET_DIR, TABLE, 1, shuffle=True, imgnrs=range(*TRAIN_RANGE))
    # traingen2 = Torch3DDataset(DATASET_DIR, TABLE, 1, shuffle=False, imgnrs=range(*TRAIN_RANGE))
    valgen = Torch3DDataset(DATASET_DIR, TABLE, 1, shuffle=False, imgnrs=range(*VAL_RANGE))

    BATCH_SIZE = 8
    trainloader = DataLoader(
        dataset = traingen,
        batch_size = BATCH_SIZE,
        collate_fn = collate,
        pin_memory = True,
        shuffle = True
    )

    valloader = DataLoader(
        dataset = valgen,
        # dataset = traingen2,
        batch_size = BATCH_SIZE,
        collate_fn = collate,
        pin_memory = True
    )

    models = [FishDETR_first, FishDETR_alt, FishDETR_split]
    notes = [
        "Regular FishDETR", 
        "Transformers takes in concatendated ResNet features", 
        "Split FC"
    ]
    for model, note in zip(models, notes):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
        optimizer.param_groups[0]['lr'] = 5e-6
        weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1, 'loss_smooth':1}
        losses = ['labels', 'boxes_3d']
        matcher = HungarianMatcher(use_giou=False, smooth_l1=False)
        criterion = SetCriterion(6, matcher, weight_dict, eos_coef = 0.5, losses=losses)
        criterion = criterion.to(device)

        train_model(
            trainloader,
            valloader,
            model,
            criterion,
            optimizer,
            n_epochs=50,
            device=device,
            validate=True,
            save_best=True,
            save_last=True,
            check_save_in_interval=1,
            weights_dir=WEIGHTS_DIR,
            notes=note
        )

