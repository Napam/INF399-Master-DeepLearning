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


def _validate_model(context: dict, traintqdminfo: dict) -> dict:
    model = context['model']
    criterion = context['criterion']

    running_val_loss = 0.0
    # valbar will disappear after it is done since leave=False
    valbar = tqdm(
        iterable=enumerate(context['valloader'], 0), 
        total=len(context['valloader']), 
        unit=' batches',
        desc=f' Validating',
        ascii=True,
        position=0,
        leave=False,
        file=sys.stdout,
        ncols=100
    )

    model.eval()
    criterion.eval()

    # Loop through val batches
    with torch.no_grad():
        for i, (images, labels) in valbar:
            X, y = detr.preprocess(images, labels, context['device'])

            output: detr.DETROutput
            loss: torch.Tensor
            output, loss = model.eval_on_batch(X, y, criterion)
            
            # print statistics
            running_val_loss += loss.item()
            val_loss = running_val_loss / (i+1)
            valtqdminfo = {**traintqdminfo, 'val loss':val_loss}
            valbar.set_postfix(valtqdminfo)
            
    return valtqdminfo, output, loss, running_val_loss / (i+1)


def _train_model(context: dict, epoch: int, n_epochs: int, leave_tqdm: bool) -> Tuple[Iterable, dict]:
    model = context['model']
    criterion = context['criterion']
    optimizer = context['optimizer']
    
    running_train_loss = 0.0
    trainbar = tqdm(
        iterable=enumerate(context['trainloader'], 0),
        total=len(context['trainloader']),
        unit=' batches',
        desc=f' Epoch {epoch+1}/{n_epochs}',
        ascii=True,
        position=0,
        leave=leave_tqdm,
        file=sys.stdout,
        ncols=100
    )

    model.train()
    criterion.train()

    # Loop through train batches
    for i, (images, labels) in trainbar:
        X, y = detr.preprocess(images, labels, context['device'])

        output: detr.DETROutput
        loss: torch.Tensor
        output, loss = model.train_on_batch(X, y, criterion, optimizer)

        # print statistics
        running_train_loss += loss.item()
        train_loss = running_train_loss / (i+1)
        traintqdminfo = {'train loss':train_loss}
        trainbar.set_postfix(traintqdminfo)
    
    return trainbar, traintqdminfo, output, loss, running_train_loss


@utils.interruptable
def train_model(
        trainloader: DataLoader, 
        valloader: DataLoader, 
        model: nn.Module, 
        criterion, 
        optimizer, 
        n_epochs: int, 
        device: torch.device, 
        validate: bool = True,
        save_best: bool = True,
        check_save_in_interval: int = 1
    ):
    
    # for convenience
    context = {
        'trainloader':trainloader,
        'valloader':valloader,
        'model':model,
        'criterion':criterion,
        'optimizer':optimizer,
        'device':device
    }
    
    best_val_loss = np.inf
    
    for epoch in range(n_epochs):
        trainbar, traintqdminfo, _, _, _ = _train_model(context, epoch, n_epochs, not validate)
            
        if validate:
            valtqdminfo, _, _, _ = _validate_model(context, traintqdminfo)
        
            # Extra dirty tqdm hack hehe
            # _validate_model will create its own tqdm bar that will replace the bar
            # from _train_model, but will clear itself afterwards
            # the code below reactivates the previous train bar
            trainbar.disable = False
            trainbar.set_postfix({**traintqdminfo, **valtqdminfo})
            trainbar.disable = True
            print() # for newline
        
            # Save best models
            if save_best:
                if valtqdminfo['val loss'] < best_val_loss:
                    best_val_loss = valtqdminfo['val loss']
                    isodatenow = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                    daydir = datetime.today().strftime("weights_%Y-%m-%d")
                    filename = (
                        f'detr_statedicts_epoch{epoch+1}'
                        f'_train{traintqdminfo["train loss"]:.4f}_val{best_val_loss:.4f}'
                        f'_{isodatenow}.pth'
                    )
                    filepath = os.path.join(WEIGHTS_DIR, daydir, filename)

                    # Save sparsely if needed
                    if not epoch % check_save_in_interval:
                        utils.save_model(
                            obj={
                                'model_state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'criterion':criterion.state_dict(),
                            },
                            f = filepath
                        )
                        print(f"\nSaved model: {filepath}\n")


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
    DATASET_DIR = '/mnt/blendervol/3d_data'
    TABLE = 'bboxes_full'
    WEIGHTS_DIR = 'fish_statedicts_3d'
    torch.hub.set_dir(TORCH_CACHE_DIR)
    num2name = eval(open(os.path.join(DATASET_DIR,"metadata.txt"), 'r').read())

    model = detr.FishDETR().to(device)
    model.load_state_dict(torch.load('fish_statedicts_3d/weights_2021-03-25/detr_statedicts_epoch13_train0.1328_val0.1285_2021-03-25T08:59:04.pth')['model_state_dict'])
    # model.load_state_dict(torch.load('last_epoch_detr_3d.pth'))

    db_con = sqlite3.connect(f'file:{os.path.join(DATASET_DIR,"bboxes.db")}?mode=ro', uri=True)
    print("Getting number of images in database")
    n_data = pd.read_sql_query(f'SELECT COUNT(DISTINCT(imgnr)) FROM {TABLE}', db_con).values[0][0]

    TRAIN_RANGE = (0, int(9/10*n_data))
    VAL_RANGE = (int(9/10*n_data), n_data)

    # TRAIN_RANGE = (0, 576)
    # VAL_RANGE = (0, 256)
    
    # TRAIN_RANGE = (0, 6*16)
    # VAL_RANGE = (384, 416)

    traingen = Torch3DDataset(DATASET_DIR, TABLE, 1, shuffle=True, imgnrs=range(*TRAIN_RANGE))
    # traingen2 = Torch3DDataset(DATASET_DIR, TABLE, 1, shuffle=False, imgnrs=range(*TRAIN_RANGE))
    valgen = Torch3DDataset(DATASET_DIR, TABLE, 1, shuffle=False, imgnrs=range(*VAL_RANGE))

    BATCH_SIZE = 8
    trainloader = DataLoader(
        dataset = traingen,
        batch_size = BATCH_SIZE,
        collate_fn = detr.collate,
        pin_memory = True,
        shuffle = True
    )

    valloader = DataLoader(
        dataset = valgen,
        # dataset = traingen2,
        batch_size = BATCH_SIZE,
        collate_fn = detr.collate,
        pin_memory = True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.param_groups[0]['lr'] = 1e-5
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
        n_epochs=20,
        device=device,
        save_best=True,
        validate=True,
        check_save_in_interval=1
    )

    utils.save_model(model.state_dict(), "last_epoch_detr_3d.pth")
    
