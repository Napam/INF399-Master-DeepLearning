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
from train_fishdetr3d import _validate_model
from tqdm import tqdm 
from typing import List, Dict
import utils


@torch.no_grad()
def get_preds_as_csvs(models_n_stuff: List[dict], context: dict):
    batch_size = context['batch_size']
    valloader = context['valloader']
    mapdir = context['mapdir']

    for dict_ in models_n_stuff:
        api = dict_['api']
        weightdir = dict_['weightdir']
        name = dict_['name']

        model = api.FishDETR(pretrained_enc=False).to(device)

        if weightdir:
            try:
                statedict = torch.load(weightdir, map_location='cpu')['model_state_dict']
            except KeyError:
                statedict = torch.load(weightdir, map_location='cpu')['model']

            model.load_state_dict(statedict)
            del statedict
        
        valbar = tqdm(
            iterable=enumerate(valloader, 0),
            total=len(valloader), 
            unit='batches',
            desc=name,
            ascii=True,
            position=0,
            leave=True,
            ncols=80
        )

        outfile = os.path.join(mapdir, f'nogit_output_{name}.csv')
        with open(outfile, 'w') as f:
            f.write('imgnr,conf,class_,x,y,z,w,l,h,rx,ry,rz\n')

        indices = valloader.dataset.indices
        for i, (images, labels) in valbar:
            X, y = preprocess(images, labels, device)
            output = model(X)
            df: pd.DataFrame = api.postprocess_to_df(indices[i*batch_size:(i+1)*batch_size], output, 0.5)
            df.to_csv(outfile, mode='a', header=None)
    

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
    VAL_RANGE = (1000, 2000)

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

    mapdir = "mapdir"
    os.makedirs(mapdir, exist_ok=True)

    models_n_stuff = [
        {'api':fishdetr3d_regular,
         'name':'regular60k_test',
         'weightdir':"fish_statedicts/weights_2021-05-31/trainsession_2021-05-31T16h26m22s/detr_statedicts_epoch10_train0.1116_val0.1060.pth"},
        
        # {'api':fishdetr3d_regular,
        #  'name':'regular25k',
        #  'weightdir':"fish_statedicts/weights_2021-05-21/trainsession_2021-05-21T08h58m00s/detr_statedicts_epoch47_train0.1166_val0.1192.pth"},
        
        # {'api':fishdetr3d_alt, 
        #  'name':'alt',
        #  'weightdir':"fish_statedicts/weights_2021-05-22/trainsession_2021-05-22T19h07m55s/detr_statedicts_epoch3_train0.2719_val0.6996.pth"},
        
        # {'api':fishdetr3d_splitfc, 
        #  'name':'splitfc',
        #  'weightdir':"fish_statedicts/weights_2021-05-25/trainsession_2021-05-25T18h09m48s/detr_statedicts_epoch40_train0.1444_val0.1449.pth"},
        
        # {'api':fishdetr3d_sincos,
        #  'name':'sincos',
        #  'weightdir':"fish_statedicts/weights_2021-05-28/trainsession_2021-05-28T08h23m37s/detr_statedicts_epoch46_train0.5501_val0.5497.pth"}
    ]

    pd.read_sql(f"""
        SELECT * FROM bboxes_full 
        WHERE imgnr IN ({','.join(map(str, valgen.indices))})
    """, valgen.con).to_csv(os.path.join(mapdir, "nogit_val_labels.csv"), index=False)

    context = {'valloader':valloader, 'batch_size':BATCH_SIZE, 'mapdir':mapdir}

    get_preds_as_csvs(models_n_stuff, context)