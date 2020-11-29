import sqlite3 as db 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset
from typing import Iterable, Optional, Sequence, Union, Callable
from torch import nn
import torch.nn.functional as F
import torch
from PIL import Image
import os 
from skimage import io

class BlenderDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        table: str,
        batch_size: int,
        n_classes: Optional[int] = None, 
        shuffle: bool=True,
        preprocessor: Optional[Callable[[torch.tensor], torch.tensor]] = None,
        *ppargs,
        **ppkwargs
        ):
        '''
        data_dir: str, path to blender generated_data directory
        table: str, name of sqlite3 table
        '''
        self.con = db.connect(f'file:{os.path.join(data_dir,"bboxes.db")}?mode=ro', uri=True)
        self.c = self.con.cursor()

        self.df: pd.DataFrame = pd.read_sql_query(f'SELECT * FROM {table}', con=self.con)
        self.n = len(self.df)

        if n_classes is None:
            n_classes = len(self.df['class_'].unique())

        # np.unique sorts by default
        self.indices = np.unique(self.df['imgnr'])
            
        if shuffle:
            np.random.shuffle(self.indices)

        self.img_dir = os.path.join(data_dir, 'images')

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.ppargs = ppargs
        self.ppkwargs = ppkwargs

    def __len__(self) -> int:
        return self.n // self.batch_size

    def _get_batch_indices(self, batchnr: int) -> Iterable:
        return self.indices[self.batch_size*batchnr:self.batch_size*(batchnr+1)]

    def _get_imgs(self): ...

    def _get_labels(self, batchnr: int) -> pd.DataFrame: 
        indices = self._get_batch_indices(batchnr)
        return self.df.query('imgnr in @indices').loc[:,'class_':]

    def get_batch(self, batchnr: int):
        '''
        Get batch
        
        If preprocess function is found in self, then things will be preprocessed
        '''
        # Get indices of current batch
        # indices correspond to imgnr
        batch_indices = self._get_batch_indices(batchnr)
        X_batch = [
            (
                # Transpose to C H W
                torch.tensor(io.imread(os.path.join(self.img_dir, f'img{i}_L.png')).transpose((2,1,0))),
                torch.tensor(io.imread(os.path.join(self.img_dir, f'img{i}_R.png')).transpose((2,1,0))),
            )
            for i in batch_indices
        ]

        # If preprocessor function is given
        if self.preprocessor is not None:
            for i in range(self.batch_size):
                X_batch[i] = self.preprocessor(X_batch[i], *self.ppargs, **self.ppkwargs)
        
        y_batch = self.df.query('imgnr in @batch_indices')[['x','y','z','class_']]
        return X_batch, y_batch

    def __getitem__(self, batchnr: int):
        return self.get_batch(batchnr)

if __name__ == '__main__': 
    thing = BlenderDataset(
        data_dir='/mnt/generated_data',
        table='bboxes_xyz',
        batch_size=2
    )

    X, y = thing.get_batch(0)

    print('X')
    print(X[0][0].shape)
    print(X[0][1].shape)
    print('y')
    print(y)
