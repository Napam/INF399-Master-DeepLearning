from typing import Any, Callable, Union
import torch
import importlib
from types import ModuleType
import random 
import os 
import numpy as np
import pathlib
from functools import wraps
import inspect
import re

def pytorch_init():
    torch.cuda.set_device(1)
    torch.cuda.current_device()
    
    # Sanity checks
    assert torch.cuda.current_device() == 1, 'Using wrong GPU'
    assert torch.cuda.device_count() == 2, 'Cannot find both GPUs'
    assert torch.cuda.get_device_name(0) == 'GeForce RTX 2080 Ti'
    assert torch.cuda.is_available() == True, 'GPU not available'  


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  


def reloader(module_or_member: Union[ModuleType, Any]):    
    if isinstance(module_or_member, ModuleType):
        importlib.reload(module_or_member)
        return module_or_member
    else:
        module = importlib.import_module(module_or_member.__module__)
        importlib.reload(module)
        return module.__dict__[module_or_member.__name__]


def get_cuda_status(device: Union[str, int, torch.device]) -> str: 
    assert device.type == 'cuda', 'device not cuda'

    mesg = f"{torch.cuda.get_device_name(device)} \n"
    mesg += "Memory usage:\n"
    mesg += f"Allocated: {round(torch.cuda.memory_allocated(device)/1024**3,1)} GB\n"
    mesg += f"Cached   : {round(torch.cuda.memory_reserved(device)/1024**3,1)} GB"
    return mesg

def save_model(obj: Any, f: str):
    pathlib.Path(f).parent.mkdir(parents=True, exist_ok=True)
    assert isinstance(f, str), "Filename must be of type string when saving model"
    torch.save(obj=obj, f=f)
    
    
def interruptable(f: Callable):
    '''Decorator for functions that should handle KeyboardInterrupts more gracefully'''
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            retval = f(*args, **kwargs)
        except KeyboardInterrupt as e:
            print('Process interrupted')
            return 
        return retval
    return wrapper

def dict_union_update(a: dict, b: dict):
    '''Updates "a" with the union of "a" and "b"'''
    a.update((                                   # Set union
        (key, b.get(key, a.get(key))) for key in a.keys() & b.keys()
    ))

def debugt(tensor: Union[np.ndarray, torch.Tensor]):
    '''
    Stands for debug tensor

    If called in methods, relies on that self is called self
    '''
    frames = inspect.stack()
    
    # frames[0] is current frame,
    # frames[1] is previous
    # frames[-1] is first frame
    # remember stacks are lifo

    funcname = frames[1].function
    lineno = frames[1].lineno
    pre_code = frames[1].code_context[0]
    arg = re.findall(r'debugt\((.*?)\)', pre_code)[0]
    
    f_locals = frames[1].frame.f_locals

    if "self" in f_locals:
        classname = f_locals["self"].__class__.__name__
        print(f'\033[32m({lineno}, {classname}.{funcname}) \033[0m{arg}: {tensor.shape}')
    else:
        print(f'\033[32m({lineno}, {funcname}) \033[0m{arg}: {tensor.shape}')