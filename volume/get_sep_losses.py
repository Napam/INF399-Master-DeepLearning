import fishdetr3d as fishdetr3d_regular
import fishdetr3d_alt
import fishdetr3d_splitfc
import fishdetr3d_sincos
from fishdetr3d_sincos import postprocess_to_df as postprocess_to_df_sincos
from fishdetr3d import collate, postprocess_to_df, preprocess
import torch
import os 
import pandas as pd 
import sqlite3
from generators import Torch3DDataset
from torch.utils.data import DataLoader
from tqdm import tqdm 
from typing import List, Dict
import utils
from setcriterion import SetCriterion
from hungarianmatcher import HungarianMatcher
import numpy as np 

def get_stuff(model, criterion, device, valloader) -> dict:
    running_val_loss = 0.0
    # valbar will disappear after it is done since leave=False
    valbar = tqdm(
        iterable=enumerate(valloader, 0), 
        total=len(valloader), 
        unit=' batches',
        desc=f' Validating',
        ascii=True,
        position=0,
        leave=False,
        ncols=100
    )

    model.eval()
    criterion.eval()
    
    loclosses = np.empty(len(valloader))
    dimlosses = np.empty(len(valloader))
    rotlosses = np.empty(len(valloader))

    # Loop through val batches
    with torch.no_grad():
        for i, (images, labels) in valbar:
            X, y = preprocess(images, labels, device)

            output = model(X)
            sep_losses = criterion(output, y)

            loclosses[i] = sep_losses['loss_loc'].cpu().item()
            dimlosses[i] = sep_losses['loss_dim'].cpu().item()
            rotlosses[i] = sep_losses['loss_rot'].cpu().item()

    print(loclosses.mean(), loclosses.std())
    print(dimlosses.mean(), dimlosses.std())
    print(rotlosses.mean(), rotlosses.std())
    

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
    TABLE = 'bboxes_full'
    torch.hub.set_dir(TORCH_CACHE_DIR)
    num2name = eval(open(os.path.join(DATASET_DIR,"metadata.txt"), 'r').read())

    db_con = sqlite3.connect(f'file:{os.path.join(DATASET_DIR,"bboxes.db")}?mode=ro', uri=True)
    print("Getting number of images in database: ", end="")
    n_data = pd.read_sql_query(f'SELECT COUNT(DISTINCT(imgnr)) FROM {TABLE}', db_con).values[0][0]
    print(n_data)

    # VAL_RANGE = (0,1000)
    VAL_RANGE = (1000,2000)
    # VAL_RANGE = (0,64)

    print(f"VAL_RANGE: {VAL_RANGE}")
    valgen = Torch3DDataset(DATASET_DIR, TABLE, 1, shuffle=False, imgnrs=range(*VAL_RANGE))

    BATCH_SIZE = 8

    valloader = DataLoader(
        dataset = valgen,
        batch_size = BATCH_SIZE,
        collate_fn = collate,
        pin_memory = False,
        shuffle=False
    )

    context = {
        'api':fishdetr3d_regular,
        'name':'regular',
        'weightdir':"fish_statedicts/weights_2021-05-31/trainsession_2021-05-31T16h26m22s/detr_statedicts_epoch10_train0.1116_val0.1060.pth"
    }

    api = context['api']
    weightdir = context['weightdir']
    name = context['name']

    model = api.FishDETR(pretrained_enc=False).to(device)

    if weightdir:
        try:
            statedict = torch.load(weightdir, map_location='cpu')['model_state_dict']
        except KeyError:
            statedict = torch.load(weightdir, map_location='cpu')['model']

        model.load_state_dict(statedict)
        del statedict

    weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1, 'loss_smooth':1}
    losses = ['labels', 'boxes_3d_sep']
    matcher = HungarianMatcher(use_giou=False, smooth_l1=False)
    criterion = SetCriterion(6, matcher, weight_dict, eos_coef = 0.5, losses=losses)
    criterion = criterion.to(device)

    get_stuff(model, criterion, device, valloader)


    # 0.0028483110312372446                                                                               
    # 0.004172523012384772
    # 0.03850884599983692

    # XTest
    # 0.0030690934965386988 0.001118043372348415
    # 0.004454518537037075 0.0018166043194410208
    # 0.0376763304322958 0.004623436958799256
