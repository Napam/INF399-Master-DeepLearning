import sqlite3 as db 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset
from typing import Dict, Iterable, Optional, Sequence, Union, Callable, List, Tuple, Generator
from torch import nn
import torch.nn.functional as F
import torch
from PIL import Image
import os 
from skimage import io
from abc import abstractmethod, ABC
from pprint import pprint
from matplotlib import pyplot as plt 
from matplotlib import patches

class BlenderDatasetBase(Dataset, ABC):
    def __init__(
            self, 
            data_dir: str,
            table: str,
            batch_size: int,
            imgnrs: Optional[Iterable[int]] = None,
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

        query = f'SELECT * FROM {table}'        
            
        if imgnrs is not None:
            query += f' WHERE imgnr IN ({",".join(str(i) for i in imgnrs)})'

        self.df: pd.DataFrame = pd.read_sql_query(query, con=self.con)        

        self.n = len(pd.unique(self.df['imgnr']))

        if n_classes is None:
            n_classes = len(self.df['class_'].unique())

        # np.unique sorts by default
        self.indices = np.unique(self.df['imgnr'])
            
        if shuffle:
            np.random.shuffle(self.indices)

        self.img_dir = os.path.join(data_dir, 'images')

        with open(os.path.join(data_dir, 'metadata.txt')) as f:
            self.num2name = eval(f.readline())

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.ppargs = ppargs
        self.ppkwargs = ppkwargs

    def __len__(self) -> int:
        return self.n // self.batch_size

    def _get_batch_indices(self, batchnr: int) -> Iterable:
        # print(f'From {self.batch_size*batchnr} to {self.batch_size*(batchnr+1)}')
        return self.indices[self.batch_size*batchnr:self.batch_size*(batchnr+1)]

    def get_image(self, path: str):
        img: np.ndarray = io.imread(path) / 255
        return img

    def get_labels(self, batchnr: int) -> pd.DataFrame: 
        indices = self._get_batch_indices(batchnr)
        return self.df.query('imgnr in @indices').loc[:,'class_':]

    @abstractmethod
    def get_batch(self, batchnr: int): ...

    def __getitem__(self, batchnr: int):
        # zero indexes, so much stop one iteration before len
        if batchnr >= len(self):
            raise StopIteration
        return self.get_batch(batchnr)


class BlenderStereoDataset(BlenderDatasetBase):
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

        super().__init__(
            data_dir, table, batch_size, n_classes, shuffle, preprocessor, *ppargs, **ppkwargs)


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


class BlenderStandardDataset(BlenderDatasetBase):
    def __init__(
        self, 
        data_dir: str,
        table: str,
        batch_size: int,
        imgnrs: Optional[Iterable[int]] = None,
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
        super().__init__(
            data_dir = data_dir,
            table = table,
            batch_size = batch_size,
            imgnrs = imgnrs,
            n_classes = n_classes,
            shuffle = shuffle,
            preprocessor = preprocessor,
            *ppargs,
            **ppkwargs
        )    


    def get_batch(self, batchnr: int) -> Tuple[List[Sequence[Sequence[Sequence[float]]]], List[Tuple[Sequence[int], Sequence[Sequence[float]]]]]:
        '''
        Get batch
        
        If preprocess function is found in self, then things will be preprocessed
        '''
        # Get indices of current batch
        # indices correspond to imgnr

        batch_indices = self._get_batch_indices(batchnr)
        X_batch: List[np.ndarray] = [
            self.get_image(os.path.join(self.img_dir, f'img{i}.png')) for i in batch_indices
        ]

        # If preprocessor function is given
        if self.preprocessor is not None:
            for i in range(self.batch_size):
                X_batch[i] = self.preprocessor(X_batch[i], *self.ppargs, **self.ppkwargs)
        
        y_batch: List[Tuple[Sequence[int], Sequence[Sequence[float]]]] = []

        for i in batch_indices:
            temp = self.df.query('imgnr == @i').values
            # 0th column: imgnr, 1st column: class_, rest is bbox
            y_batch.append((temp[:,1].astype(int), temp[:,2:]))

        return X_batch, y_batch

    def plot_bbox(self, img, bboxes, ax):
        img_h, img_w = img.shape[:-1]

        for class_, (x, y, w, h) in zip(*bboxes):
            topLeftCorner = (x*img_h, y*img_w)
            ax.add_patch(patches.Rectangle(
                topLeftCorner, 
                w*img_w, 
                h*img_h, 
                facecolor='none', 
                edgecolor='red', 
                linewidth=2,
                alpha=0.4
            ))
            ax.text(
                x=topLeftCorner[0], 
                y=topLeftCorner[1], 
                s=self.num2name[int(class_)], 
                bbox=dict(facecolor='red', alpha=0.4, linewidth=2, edgecolor='none'), 
                color='w',
                fontsize=10
            )

    def plot_batch(self, batchnr: int):
        imgs, bboxes = self.get_batch(batchnr)
        # print(imgs)
        fig, axes = plt.subplots(1, len(imgs), figsize=(7,5))

        for img, bboxes_for_img, ax in zip(imgs, bboxes, np.ravel(axes)):
            ax.imshow(img)
            self.plot_bbox(img, bboxes_for_img, ax)


class TorchStandardDataset(BlenderStandardDataset):
    def __init__(
            self, 
            data_dir: str,
            table: str,
            batch_size: Optional[int] = None, # Unused
            imgnrs: Optional[Iterable[int]] = None,
            n_classes: Optional[int] = None, 
            shuffle: bool=True,
            preprocessor: Optional[Callable[[torch.tensor], torch.tensor]] = None,
            device: Optional[torch.device] = None,
            *ppargs,
            **ppkwargs
        ):
        '''
        data_dir: str, path to blender generated_data directory
        table: str, name of sqlite3 table
        '''
        super().__init__(
            data_dir = data_dir,
            table = table,
            batch_size = 1,
            imgnrs = imgnrs,
            n_classes = n_classes,
            shuffle = shuffle,
            preprocessor = preprocessor,
            *ppargs,
            **ppkwargs
        )    
        self.device = device    

    def get_batch(self, batchnr: int) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        X_batch, y_batch = super().get_batch(batchnr)

        X_batch: torch.Tensor = torch.as_tensor(X_batch[0], device=self.device, dtype=torch.float32).permute((2,0,1)) 
        
        # None device will default to CPU or something (it doesn't crash hehe)
        y_batch: List[Dict[str, torch.tensor]] = {
                'labels': torch.as_tensor(y_batch[0][0], dtype=torch.long, device=self.device), 
                'boxes': torch.as_tensor(y_batch[0][1], dtype=torch.float32, device=self.device)
            } 
        
        return X_batch, y_batch


    def plot_batch(self, batchnr: int):
        (img,), (bboxes,) = super().get_batch(batchnr)
        plt.imshow(img)
        self.plot_bbox(img, bboxes, plt.gca())
            

if __name__ == '__main__': 
    thing = BlenderStandardDataset(
        data_dir='/mnt/blendervol/objdet_std_data',
        table='bboxes_std',
        batch_size=4,
        imgnrs=(1,1,1)
    )

    # X, y = thing.get_batch(0)
    # pprint(y)

    # print('X')
    # print(X[0][0].shape)
    # print(X[0][1].shape)
    # print('y')
    # print(y)
    pass
