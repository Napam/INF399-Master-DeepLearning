from unicodedata import name
import numpy as np
import os 
from datetime import datetime
import pathlib
import pandas as pd 
from matplotlib import pyplot as plt 
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset':'dejavuserif'})


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


def lossplots_combine(name_n_paths: list, compare: bool=False):
    for stuff in name_n_paths:
        name = stuff[0]
        paths = stuff[1:]
        
        df = pd.concat((pd.read_csv(pathlib.Path(path) / "losses.csv") for path in paths))
        # print(df.train.values)
        # exit()
        plt.plot(df.train.values, c='k', label='train', linewidth=1)
        plt.plot(df.val.values, c='orangered', label='val', linewidth=1)
        plt.axvline(99, linewidth=0.8, linestyle='--', c='gray')
        plt.title(name)
        plt.legend()
        plt.savefig("plots/"+f"lossplot_{name}.png")
        plt.savefig("plots/"+f"lossplot_{name}.pdf")
        plt.close()
    
    if compare:
        for stuff in name_n_paths:
            name = stuff[0]
            paths = stuff[1:]
            df = pd.concat((pd.read_csv(pathlib.Path(path) / "losses.csv") for path in paths))
            plot = plt.plot(df.val.values, label=f"{name} (Val)", linewidth=1)
            plt.plot(df.train.values, label=f"{name} (Train)", linewidth=0.8, c=plot[0].get_color(), linestyle='--')
            plt.axvline(99, linewidth=0.8, linestyle='--', c='gray')

        plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
            fancybox=True, shadow=False, ncol=2)
        plt.title("Loss comparison of models")
        plt.savefig("plots/"+f"compare.png", bbox_inches='tight')
        plt.savefig("plots/"+f"compare.pdf", bbox_inches='tight')
        plt.close()

def lossplot_mergefish60k(name_n_paths: list):
    for stuff in name_n_paths:
        name = stuff[0]
        paths = stuff[1:]
        
        df = pd.concat((pd.read_csv(pathlib.Path(path) / "losses.csv") for path in paths))
        print(len(df))
        # print(df.train.values)
        # exit()
        plt.plot(df.train.values, c='k', label='train', linewidth=1)
        plt.plot(df.val.values, c='orangered', label='val', linewidth=1)
        plt.axvline(99, linewidth=0.8, linestyle='--', c='gray')
        plt.axvline(149, linewidth=0.8, linestyle='--', c='gray')
        plt.title(name)
        plt.legend()
        plt.savefig("plots/"+f"lossplot_{name}.png")
        plt.savefig("plots/"+f"lossplot_{name}.pdf")
        plt.close()



if __name__ == '__main__':
    name_n_paths_comb = [
        ('MergeFish3D60K',
         'fish_statedicts/weights_2021-05-16/trainsession_2021-05-16T17h19m11s',
         'fish_statedicts/weights_2021-05-21/trainsession_2021-05-21T08h58m00s',
         'fish_statedicts/weights_2021-05-29/trainsession_2021-05-29T15h56m11s',
         'fish_statedicts/weights_2021-05-30/trainsession_2021-05-30T09h41m36s',
         'fish_statedicts/weights_2021-05-30/trainsession_2021-05-30T15h15m21s',
         'fish_statedicts/weights_2021-05-31/trainsession_2021-05-31T16h26m22s'),
        
        # ('CatFish3D',
        #  'fish_statedicts/weights_2021-05-18/trainsession_2021-05-18T01h58m16s',
        #  'fish_statedicts/weights_2021-05-22/trainsession_2021-05-22T19h07m55s/'),
        
        # ('SplitFish3D',
        #  'fish_statedicts/weights_2021-05-19/trainsession_2021-05-19T07h30m51s',
        #  'fish_statedicts/weights_2021-05-25/trainsession_2021-05-25T18h09m48s'),
        
        # ('SinusFish3D',
        #  'fish_statedicts/weights_2021-05-27/trainsession_2021-05-27T07h33m16s',
        #  'fish_statedicts/weights_2021-05-28/trainsession_2021-05-28T08h23m37s',
        #  'fish_statedicts/weights_2021-05-29/trainsession_2021-05-29T10h06m13s'),
    ]

    # lossplots_combine(name_n_paths_comb)
    lossplot_mergefish60k(name_n_paths_comb)