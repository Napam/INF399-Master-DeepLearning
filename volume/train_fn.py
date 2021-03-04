from typing import Generic, Optional, Tuple, List, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import utils
from utils import debugt, debugs, debug
from datetime import datetime
from torch import optim
import gc

import fishdetr_batchboy as detr

# import detr_batchboy_regular as detr
from generators import TorchStereoDataset
from tqdm import tqdm

import sys

sys.path.append("./detr_custom/")
from models.matcher import HungarianMatcher
from models.detr import SetCriterion
import os
import sqlite3
from queue import Queue
import io

seed = 42069
utils.seed_everything(seed)


TORCH_CACHE_DIR = "torch_cache"
DATASET_DIR = "/mnt/blendervol/leftright_left_data"
TABLE = "bboxes_std"
WEIGHTS_DIR = "fish_statedicts"
torch.hub.set_dir(TORCH_CACHE_DIR)
num2name = eval(open(os.path.join(DATASET_DIR, "metadata.txt"), "r").read())


def _validate_model(context: dict, traintqdminfo: dict) -> dict:
    model = context["model"]
    criterion = context["criterion"]
    stop_callback = context["stop_callback"]

    running_val_loss = 0.0
    # valbar will disappear after it is done since leave=False
    valbar = tqdm(
        iterable=enumerate(context["valloader"], 0),
        total=len(context["valloader"]),
        unit=" batches",
        desc=f" Validating",
        ascii=True,
        position=0,
        leave=False,
        file=context["file"],
    )

    model.eval()
    criterion.eval()

    # Loop through val batches
    with torch.no_grad():
        for i, (images, labels) in valbar:
            stop_callback()
            X, y = detr.preprocess(images, labels, context["device"])

            output: detr.DETROutput
            loss: torch.Tensor
            output, loss = model.eval_on_batch(X, y, criterion)

            # print statistics
            running_val_loss += loss.item()
            val_loss = running_val_loss / (i + 1)
            valtqdminfo = {**traintqdminfo, "val loss": val_loss}
            valbar.set_postfix(valtqdminfo)

    return valtqdminfo


def _train_model(context: dict, leave_tqdm: bool) -> Tuple[Iterable, dict]:
    model: nn.Module = context["model"]
    criterion: nn.Module = context["criterion"]
    optimizer: optim.Optimizer = context["optimizer"]
    n_epochs: int = context["n_epochs"]
    epoch: int = context["epoch"]
    batch_callbacks: Queue = context["batch_callbacks"]
    stop_callback = context["stop_callback"]

    running_train_loss = 0.0
    trainbar = tqdm(
        iterable=enumerate(context["trainloader"], 0),
        total=len(context["trainloader"]),
        unit=" batches",
        desc=f" Epoch {epoch+1}/{n_epochs}",
        ascii=True,
        position=0,
        leave=leave_tqdm,
        file=context["file"],
    )
    context["n_batches"] = len(context["trainloader"])

    model.train()
    criterion.train()

    # Loop through train batches
    for batch, (images, labels) in trainbar:
        context["batch"] = batch
        stop_callback()
        X, y = detr.preprocess(images, labels, context["device"])

        output: detr.DETROutput
        loss: torch.Tensor
        output, loss = model.train_on_batch(X, y, criterion, optimizer)

        # print statistics
        running_train_loss += loss.item()
        train_loss = running_train_loss / (batch + 1)
        traintqdminfo = {"train loss": train_loss}
        trainbar.set_postfix(traintqdminfo)

        while not batch_callbacks.empty():
            batch_callbacks.get()(context)

    return trainbar, traintqdminfo


# @utils.interruptable
def train_model(
    trainloader: DataLoader,
    valloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    n_epochs: int,
    device: torch.device,
    epoch_callbacks: Queue,
    batch_callbacks: Queue,
    validate: bool = True,
    save_best: bool = True,
    file: Optional[io.TextIOBase] = None,
    stop_callback: Optional[Callable] = None,
):

    # for convenience
    context = {
        "trainloader": trainloader,
        "valloader": valloader,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "n_epochs": n_epochs,
        "n_batches": None,
        "device": device,
        "batch_callbacks": batch_callbacks,
        "epoch_callbacks": epoch_callbacks,
        "epoch": None,  # Current epoch
        "batch": None,  # Current batch
        "file": file,
        "stop_callback": stop_callback,
    }

    if file is None:
        file = sys.stdout
        context["file"] = sys.stdout

    if stop_callback is None:
        stop_callback = lambda: None
        context["stop_callback"] = stop_callback

    best_val_loss = np.inf

    for epoch in range(n_epochs):
        context["epoch"] = epoch
        trainbar, traintqdminfo = _train_model(context, not validate)

        while not epoch_callbacks.empty():
            epoch_callbacks.get()(context)

        if validate:
            valtqdminfo = _validate_model(context, traintqdminfo)

            # Extra dirty tqdm hack hehe
            # _validate_model will create its own tqdm bar that will replace the bar
            # from _train_model, but will clear itself afterwards
            # the code below reactivates the previous train bar
            trainbar.disable = False
            trainbar.set_postfix({**traintqdminfo, **valtqdminfo})
            trainbar.disable = True
            print(file=file)  # for newline

            # Save best models
            if save_best:
                if valtqdminfo["val loss"] < best_val_loss:
                    best_val_loss = valtqdminfo["val loss"]
                    isodatenow = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                    daydir = datetime.today().strftime("weights_%Y-%m-%d")
                    filename = (
                        f"detr_statedicts_epoch{epoch+1}"
                        f'_train{traintqdminfo["train loss"]:.4f}_val{best_val_loss:.4f}'
                        f"_{isodatenow}.pth"
                    )
                    filepath = os.path.join(WEIGHTS_DIR, daydir, filename)

                    utils.save_model(
                        obj={
                            "model_state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "criterion": criterion.state_dict(),
                        },
                        f=filepath,
                    )


def train_fn(
    batch_callback_queue: Queue,
    epoch_callback_queue: Queue,
    stop_callback: Callable,
    file: io.TextIOBase,
):
    try:
        device = utils.pytorch_init_janus_gpu(0)
        print(f"Using device: {device} ({torch.cuda.get_device_name()})", file=file)
        print(utils.get_cuda_status(device), file=file)
    except AssertionError as e:
        print("GPU could not initialize, got error:", e, file=file)
        device = torch.device("cpu")
        print("Device is set to CPU", file=file)

    model = detr.FishDETR().to(device)
    model.load_state_dict(torch.load("last_epoch_detr.pth"))
    db_con = sqlite3.connect(f'file:{os.path.join(DATASET_DIR,"bboxes.db")}?mode=ro', uri=True)
    n_data = pd.read_sql_query("SELECT COUNT(DISTINCT(imgnr)) FROM bboxes_std", db_con).values[0][0]
    # TRAIN_RANGE = (0, int(3/4*n_data))
    # VAL_RANGE = (int(3/4*n_data), n_data)
    TRAIN_RANGE = (0, 12)
    VAL_RANGE = (12, 18)
    traingen = TorchStereoDataset(DATASET_DIR, TABLE, 1, shuffle=True, imgnrs=range(*TRAIN_RANGE))
    valgen = TorchStereoDataset(DATASET_DIR, TABLE, 1, shuffle=False, imgnrs=range(*VAL_RANGE))
    BATCH_SIZE = 6
    trainloader = DataLoader(
        dataset=traingen,
        batch_size=BATCH_SIZE,
        collate_fn=detr.collate,
        pin_memory=True,
    )
    valloader = DataLoader(
        dataset=valgen, batch_size=BATCH_SIZE, collate_fn=detr.collate, pin_memory=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.param_groups[0]["lr"] = 5e-6
    weight_dict = {"loss_ce": 1, "loss_bbox": 1, "loss_giou": 1}
    losses = ["labels", "boxes", "cardinality"]
    matcher = HungarianMatcher()
    criterion = SetCriterion(6, matcher, weight_dict, eos_coef=0.5, losses=losses)
    criterion = criterion.to(device)

    train_model(
        trainloader=trainloader,
        valloader=valloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=5,
        device=device,
        epoch_callbacks=epoch_callback_queue,
        batch_callbacks=batch_callback_queue,
        validate=True,
        save_best=True,
        file=file,
        stop_callback=stop_callback,
    )

    del model
    del criterion
    del matcher
    del optimizer
    del trainloader
    del traingen
    del valgen
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    return