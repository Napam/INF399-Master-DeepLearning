import numpy as np
from numpy.lib.stride_tricks import as_strided 
import torch
from utils import debug, debugs, debugt

def fixed_single_channel_2d_conv():
    a = np.arange(5*5).reshape(5,5)
    u = np.array(a.itemsize)
    b = as_strided(a, shape=(4,4,2,2), strides=u*(5,1,5,1)) 
    c = np.ones((2,2))
    print(a)
    print(np.tensordot(b, c, axes=[[2,3],[0,1]]))


def fixed_rgb_channel_2d_conv():
    a = np.arange(3*5*5).reshape(3,5,5)
    u = np.array(a.itemsize)
    b = as_strided(a, shape=(4,4,3,2,2), strides=u*(5,1,25,5,1))
    c = np.array([
        [[ 1, 1],[ 1, 1]],
        [[ 0, 0],[ 0, 0]],
        [[ 1, 1],[ 1, 1]],
    ])
    # print(a)
    print(np.tensordot(b, c, axes=[[2,3,4],[0,1,2]]))


def fixed_rgb_channel_2d_conv_torch():
    a = torch.arange(3*5*5).reshape(3,5,5)
    b = torch.as_strided(a, size=(4,4,3,2,2), stride=(5,1,25,5,1))
    c = torch.tensor([
        [[ 1, 1],[ 1, 1]],
        [[ 0, 0],[ 0, 0]],
        [[ 1, 1],[ 1, 1]],
    ])
    # print(a)
    print(torch.tensordot(b, c, dims=[[2,3,4],[0,1,2]]))


def fixed_batch_rgb_2d_conv():
    s = (2,3,5,5)
    a = np.arange(np.prod(s)).reshape(s)
    u = np.array(a.itemsize)
    b = as_strided(a, shape=(2,4,4,3,2,2), strides=u*(3*5*5,5,1,25,5,1))
    c = np.array([
        [[ 1, 1],[ 1, 1]],
        [[ 0, 0],[ 0, 0]],
        [[ 1, 1],[ 1, 1]],
    ])
    # print(a)
    print(np.tensordot(b, c, axes=[[3,4,5],[0,1,2]]))


def batch_rgb_2d_conv(a: np.ndarray, f: np.ndarray):
    n, c, h, w = a.shape
    i, k = f.shape[-2:]
    u = np.array(a.itemsize)
    b = as_strided(a, shape=(n,h-i+1,w-k+1,c,i,k), strides=u*(c*h*w,w,1,h*w,w,1))
    print(np.tensordot(b, f, axes=3))

s = (2,3,5,5)
a = np.arange(np.prod(s)).reshape(s)
k = np.array([
        [[ 1, 1],[ 1, 1]],
        [[ 0, 0],[ 0, 0]],
        [[ 1, 1],[ 1, 1]],
    ])

# k = np.stack((k,k))
batch_rgb_2d_conv(a, k)
# fixed_batch_rgb_2d_conv()
    

