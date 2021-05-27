from unicodedata import name
import numpy as np
import os 
from datetime import datetime
import pathlib
import pandas as pd 
from matplotlib import pyplot as plt 


def traintime_from_trainsession(dir_: str) -> float:
    '''
    Returns hours 
    '''
    sesspath = pathlib.Path(dir_)
    metadate = datetime.fromtimestamp((sesspath / 'metadata.txt').stat().st_mtime)
    lastdate = datetime.fromtimestamp((sesspath / 'last_epoch.pth').stat().st_mtime)
    delta = lastdate - metadate
    return delta.days*24 + delta.seconds / 60 / 60


def traintimes(name_n_paths: list):
    for name, path in name_n_paths:
        time = traintime_from_trainsession(path)
        print(f"{name:<10}: {time}")


def lossplots_combine(name_n_paths: list):
    for stuff in name_n_paths:
        name = stuff[0]
        paths = stuff[1:]
        
        df = pd.concat((pd.read_csv(pathlib.Path(path) / "losses.csv") for path in paths))
        # print(df.train.values)
        # exit()
        plt.plot(df.train.values, c='k', label='train')
        plt.plot(df.val.values, c='orangered', label='val')
        plt.title(name)
        plt.legend()
        plt.savefig(f"lossplot_{name}.png")
        plt.close()


if __name__ == '__main__':
    name_n_paths_comb = [
        ('regular',
         'fish_statedicts/weights_2021-05-16/trainsession_2021-05-16T17h19m11s/',
         'fish_statedicts/weights_2021-05-21/trainsession_2021-05-21T08h58m00s/'),
        
        ('alt',
         'fish_statedicts/weights_2021-05-18/trainsession_2021-05-18T01h58m16s',
         'fish_statedicts/weights_2021-05-22/trainsession_2021-05-22T19h07m55s/'),
        
        ('splitfc',
         'fish_statedicts/weights_2021-05-19/trainsession_2021-05-19T07h30m51s/',
         'fish_statedicts/weights_2021-05-25/trainsession_2021-05-25T18h09m48s/'),
    ]

    lossplots_combine(name_n_paths_comb)