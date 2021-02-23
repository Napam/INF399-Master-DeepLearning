from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Set, Union
import torch
from torch import nn
import importlib
from types import ModuleType
import random
import os
import numpy as np
import pathlib
from functools import wraps
import inspect
import re
from pprint import pprint
import io
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from torch.types import Number
from matplotlib.axes import Axes


def pytorch_init_janus_gpu():
    device_id = 1
    torch.cuda.set_device(device_id)

    # Sanity checks
    assert torch.cuda.current_device() == 1, "Using wrong GPU"
    assert torch.cuda.device_count() == 2, "Cannot find both GPUs"
    assert torch.cuda.get_device_name(0) == "GeForce RTX 2080 Ti", "Unexpected GPU name"
    assert torch.cuda.is_available() == True, "GPU not available"
    return torch.device("cuda", device_id)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    assert device.type == "cuda", "device not cuda"

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
    """Decorator for functions that should handle KeyboardInterrupts more gracefully"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            retval = f(*args, **kwargs)
        except KeyboardInterrupt as e:
            print("Process interrupted")
            return
        return retval

    return wrapper


def dict_union_update(a: dict, b: dict):
    '''Updates "a" with the union of "a" and "b"'''
    a.update(((key, b.get(key, a.get(key))) for key in a.keys() & b.keys()))  # Set union


def get_unique_modules(model: nn.Module) -> Set[str]:
    """
    Goes through all module and its submodules and retrieves all unique types of modules
    """
    names = set()
    model.apply(lambda x: names.add(x.__class__.__name__))
    return names


def box_cxcywh_to_xyxy(x: torch.Tensor):
    """
    Change bounding box format:

          w                 w
      ---------         x--------
      |       |         |       |
    h |   x   |  -->  h |       |
      |       |         |       |
      ---------         ---------
    """
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def box_cxcywh_to_xywh(x: torch.Tensor):
    """
    Change bounding box format:

          w                 w    
      ---------         x--------
      |       |         |       |
    h |   x   |  -->  h |       |
      |       |         |       |
      ---------         ---------
    """
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]
    return torch.stack(b, dim=1)


def plot_bboxes(
    img: ArrayLike,
    classes: Iterable,
    boxes: Iterable,
    classmap: Optional[Mapping[int, str]] = None,
    ax: Optional[Axes] = None,
    figsize: Optional[Sequence[float]] = None,
    rkwargs: Optional[Dict[str, Any]] = None,
    tkwargs: Optional[Dict[str, Any]] = None,
) -> Axes:
    """Used to plot result of 2D object detection

    Parameters
    ----------
    img : ArrayLike
        image
    classes : Iterable
        Iterable of classes
    boxes : Iterable
        Iterable of bounding boxes in xywh style (top left corner, and w and height):
              w    
          ---------
          |       |
        h |   x   |
          |       |
          ---------
    classmap : Optional[Mapping[int, str]], optional
        mapping of integer class to string class, by default None
    ax : Optional[Any], optional
        plt axes, by default None
    figsize : Optional[Tuple[float, float]], optional
        by default None
    rkwargs : Optional[Dict[str, Any]], optional
        rectangle kwargs (the bounding box), by default None
    tkwargs : Optional[Dict[str, Any]], optional
        text kwargs (the text label), by default None

    Returns
    -------
    ax: Axes
    """
    if figsize is None:
        figsize = (5, 5)

    rectkwargs = {"fill": False, "color": "cyan", "linewidth": 3}
    if rkwargs is not None:
        rectkwargs.update(rkwargs)

    textkwargs = {"fontsize": 11, "bbox": {"facecolor": "cyan", "alpha": 0.9}}
    if tkwargs is not None:
        textkwargs.update(tkwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    img = np.array(img)
    ax.imshow(img.clip(0, 1))

    if len(boxes) != 0:
        h, w = img.shape[:2]
        boxes = box_cxcywh_to_xywh(boxes)
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

        for cls, (xmin, ymin, bw, bh) in zip(classes, boxes):
            ax.add_patch(plt.Rectangle((xmin, ymin), bw, bh, **rectkwargs))
            try:
                strcls = classmap[int(cls)]
            except:
                strcls = str(int(cls))

            ax.text(xmin, ymin, strcls, **textkwargs)

    if ax is None:
        ax.axis("off")
        plt.show()

    return ax


def _tag(fname: str, offset: int = 0) -> str:
    """
    Obtain tag: (linenumber, function)
    fname: funcname
    offset: stack offset, stack frame index is 1 by default (prev stack)
    """
    frames = inspect.stack()

    # frames[0] is current frame,
    # frames[1] is previous
    # frames[-1] is first frame
    # remember stacks are LIFO
    # index 2 means 2+1 stack frames back

    frame = frames[1 + offset]
    funcname = frame.function
    lineno = frame.lineno
    pre_code = frame.code_context[0]
    arg = re.findall(rf"{fname}\((.*)\)", pre_code)[0]

    f_locals = frame.frame.f_locals

    tags = [lineno, funcname]
    if "self" in f_locals:  # Relies on convention
        classname = f_locals["self"].__class__.__name__
        tags[1] = classname + "." + funcname

    tags = f"\033[32;2m{tuple(tags)}\033[0m ".replace("'", "")

    return tags + f"\033[2m{arg}:\033[0m"


def debug(obj: Any, pretty: bool = False):
    """
    Tag and print any Python object
    """
    tag = _tag("debug", 1)
    strobj = str(obj)
    if "\n" in strobj:
        # so __str__ representation dont get wrecked if it
        # spans multiple lines
        strobj = "\n" + strobj
        strobj = strobj.replace("\n", "\n\t")
    print(f"{tag} ", end="")
    if pretty:
        with io.StringIO() as f:
            pprint(obj, stream=f)
            f.seek(0)
            print(("\n" + f.read()).replace("\n", "\n\t").rstrip())
    else:
        print(strobj)


def debugs(tensor: Any):
    """
    Tag and print shape of thing that has .shape (torch tensors, ndarrays, tensorflow tensors, ...)
    """
    tag = _tag("debugs", 1)
    print(f"{tag} {str(tensor.shape)}")


def debugt(obj: Any):
    """
    Tag and print type, if has __len__, print that too
    """
    tag = _tag("debugt", 1)
    info = f"{tag} {type(obj)}"
    try:
        info += f", len: {len(obj)}"
    except TypeError:
        pass
    print(info)
