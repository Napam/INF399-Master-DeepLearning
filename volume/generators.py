'''
IMPORTANT INFO!!!

DETR is made to output [class, (center_x, center_y, width, height)]!!!!!!!

WHILE BLENDER DATASET IS (AT THIS TIME) [class, (topLeftX, topLeftY, width, height)]!!!!!!!!!!!!!
'''

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


def np_to_torch_img(img: np.ndarray):
    # np.ndarray[H, W, C] --> torch.Tensor[C,H,W]
    return torch.as_tensor(img, dtype=torch.float32).permute((2,0,1))


class BlenderDatasetBase(Dataset, ABC):
    def __init__(self, 
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

    def get_batch_images(self, batch_indices: Iterable):
        return [
            self.get_image(os.path.join(self.img_dir, f'img{i}.png')) for i in batch_indices
        ]

    def get_batch(self, batchnr: int) -> Tuple[List[Tuple[Sequence[Sequence[Sequence[float]]]]], List[Tuple[Sequence[int], Sequence[Sequence[float]]]]]:
        '''
        Get batch
        
        If preprocess function is found in self, then things will be preprocessed
        '''
        # Get indices of current batch
        # indices correspond to imgnr

        batch_indices: Iterable = self._get_batch_indices(batchnr)
        X_batch: List[np.ndarray] = self.get_batch_images(batch_indices)

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

    def plot_batch(self, batchnr: int, figsize: Optional[Sequence[int]] = None):
        imgs, bboxes = self.get_batch(batchnr)
        # print(imgs)
        if figsize is None:
            figsize = (7,5)

        fig, axes = plt.subplots(1, len(imgs), figsize=figsize)

        for img, bboxes_for_img, ax in zip(imgs, bboxes, np.ravel(axes)):
            ax.imshow(img)
            self.plot_bbox(img, bboxes_for_img, ax)


class BlenderStereoDataset(BlenderStandardDataset):
    def __init__(self,
            data_dir: str,
            table: str,
            batch_size: int,
            imgnrs: Optional[Iterable[int]]=None,
            n_classes: Optional[int]=None,
            shuffle: bool=True,
            preprocessor: Optional[Callable[[torch.tensor], torch.tensor]]=None,
            *ppargs,
            **ppkwargs
        ):

        super().__init__(
            data_dir, 
            table, 
            batch_size, 
            imgnrs=imgnrs, 
            n_classes=n_classes, 
            shuffle=shuffle, 
            preprocessor=preprocessor, 
            *ppargs, 
            **ppkwargs
        )

    def get_batch_images(self, batch_indices: int) -> List[Tuple[np.ndarray]]:
        return [
            (
                self.get_image(os.path.join(self.img_dir, f'img{i}_L.png')),
                self.get_image(os.path.join(self.img_dir, f'img{i}_R.png'))
            ) for i in batch_indices
        ]

    def plot_batch(self, batchnr: int, figsize: Optional[Sequence[int]] = None):
        imgs, bboxes = self.get_batch(batchnr)
        # Pick out left images
        imgs = [leftright[0] for leftright in imgs]

        if figsize is None:
            figsize = (7,5)

        fig, axes = plt.subplots(1, len(imgs), figsize=figsize)

        for img, bboxes_for_img, ax in zip(imgs, bboxes, np.ravel(axes)):
            ax.imshow(img)
            self.plot_bbox(img, bboxes_for_img, ax)


class TorchStandardDataset(BlenderStandardDataset):
    def __init__(self, 
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

        X_batch: torch.Tensor = np_to_torch_img(X_batch[0]).to(self.device)
        
        # None device will default to CPU or something (it doesn't crash hehe)
        y_batch: Dict[str, torch.Tensor] = {
                'labels': torch.as_tensor(y_batch[0][0], dtype=torch.long, device=self.device), 
                'boxes': torch.as_tensor(y_batch[0][1], dtype=torch.float32, device=self.device)
            } 

        # Monkey patch hehe
        # tlx, tly, w, h (tl = top left) ---> cx, cy, w, h
        boxestensor: torch.Tensor = y_batch['boxes']
        boxestensor[:,0] = boxestensor[:,0] + boxestensor[:,2] * 0.5 # x + w / 2
        boxestensor[:,1] = boxestensor[:,1] + boxestensor[:,3] * 0.5 # y + h / 2
        
        return X_batch, y_batch


    def plot_batch(self, batchnr: int):
        (img,), (bboxes,) = super().get_batch(batchnr)
        plt.imshow(img)
        self.plot_bbox(img, bboxes, plt.gca())
            

class TorchStereoDataset(BlenderStereoDataset):
    def __init__(self, 
        data_dir: str,
        table: str,
        batch_size: int = 1, # not used
        imgnrs: Optional[Iterable[int]] = None,
        n_classes: Optional[int] =  None,
        shuffle: bool = True,
        preprocessor: Optional[Callable[[torch.tensor], torch.tensor]] = None, 
        device: Optional[torch.device] = None,
        *ppargs, 
        **ppkwargs
    ):
        super().__init__(
            data_dir, 
            table, 
            1, 
            imgnrs=imgnrs, 
            n_classes=n_classes, 
            shuffle=shuffle, 
            preprocessor=preprocessor, 
            *ppargs, 
            **ppkwargs
        )
        self.device = device
    
    def get_batch(self, batchnr: int) -> Tuple[List[Sequence[Sequence[Sequence[float]]]], List[Tuple[Sequence[int], Sequence[Sequence[float]]]]]:
        X_batch, y_batch =  super().get_batch(batchnr)
        
        # List of length 1 of tuples that contains left and right images (H, W, C)
        X_batch: List[Tuple[np.ndarray]]
        X_batch: Tuple[np.ndarray] = X_batch[0]
        X_batch: Tuple[torch.Tensor] = (np_to_torch_img(X_batch[0]).unsqueeze(0).to(self.device), 
                                        np_to_torch_img(X_batch[1]).unsqueeze(0).to(self.device))

        # None device will default to CPU or something (it doesn't crash hehe)
        y_batch: List[Dict[str, torch.Tensor]] = {
                'labels': torch.as_tensor(y_batch[0][0], dtype=torch.long, device=self.device), 
                'boxes': torch.as_tensor(y_batch[0][1], dtype=torch.float32, device=self.device)
            } 

        # Monkey patch hehe
        # tlx, tly, w, h (tl = top left) ---> cx, cy, w, h
        boxestensor: torch.Tensor = y_batch['boxes']
        boxestensor[:,0] = boxestensor[:,0] + boxestensor[:,2] * 0.5 # x + w / 2
        boxestensor[:,1] = boxestensor[:,1] + boxestensor[:,3] * 0.5 # y + h / 2
        return X_batch, y_batch

    def plot_batch(self, batchnr: int):
        (img,), (bboxes,) = super().get_batch(batchnr)
        img = img[0]
        plt.imshow(img)
        self.plot_bbox(img, bboxes, plt.gca())

if __name__ == '__main__': 
    thing = TorchStereoDataset(
        data_dir='/mnt/blendervol/leftright_left_data',
        table='bboxes_std',
        batch_size=1,
        imgnrs=range(0,1000)
    )
    
    X, y = thing.get_batch(0)

    print(X)
    print(y)

    # thing = BlenderStandardDataset(
    #     data_dir='/mnt/blendervol/objdet_std_data',
    #     table='bboxes_std',
    #     batch_size=4,
    #     imgnrs=(1,1,1)
    # )

    # X, y = thing.get_batch(0)
    # pprint(y)

    # print('X')
    # print(X[0][0].shape)
    # print(X[0][1].shape)
    # print('y')
    # print(y)
    pass
