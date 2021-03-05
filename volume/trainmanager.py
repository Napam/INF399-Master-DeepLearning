#!/usr/bin/env python

from typing import Any, Callable, Optional
from pypatconsole import menu
import threading
import time
from queue import Queue
from collections import deque

import utils
import io
from functools import wraps
import tqdm
import argparse
import importlib

# import train_fn
# import train_fn as tf
import sys
import multiprocessing
import contextlib


def are_you_sure(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f"Are you sure you want to execute {f.__name__}? y/n")
        inp = input()
        if inp == "y":
            output = f(*args, **kwargs)
            return output

    return wrapper


def call_clbks(
    callbacks: Queue, pop: bool = False, mutex: Optional[threading.Lock] = None, *args, **kwargs
):
    if mutex is None:
        mutex = contextlib.nullcontext()

    current_callback = None
    with mutex:
        try:
            if pop:
                while not callbacks.empty():
                    current_callback = callbacks.get()
                    current_callback(*args, **kwargs)
            else:
                for callback in callbacks.queue:
                    current_callback = callback
                    current_callback(*args, **kwargs)
        except Exception as e:
            print(f"\nCallback {current_callback} produced error: {e}")


def train_fn(
    batch_single_clbks: Queue,
    epoch_single_clbks: Queue,
    batch_persist_clbks: Queue,
    epoch_persist_clbks: Queue,
    stop_callback: Callable,
    file: io.BufferedIOBase,
    mutex: threading.Lock,
):
    context = {
        "n_epochs": 1000,
        "n_batches": None,
        "epoch": None,
        "batch": None,
        "epoch_single_clbks": epoch_single_clbks,
        "batch_single_clbks": batch_single_clbks,
        "batch_persist_clbks": batch_persist_clbks,
        "epoch_persist_clbks": epoch_persist_clbks,
        "file": file,
        "stop_callback": stop_callback,
    }

    for epoch in range(context["n_epochs"]):
        batches = tqdm.trange(10, file=file, ascii=True, desc="Test")
        context["epoch"] = epoch
        context["n_batches"] = len(batches)
        for batch in batches:
            stop_callback()
            time.sleep(0.5)
            call_clbks(batch_persist_clbks, pop=False, mutex=mutex, context=context)
            call_clbks(batch_single_clbks, pop=True, mutex=mutex, context=context)
        call_clbks(epoch_persist_clbks, pop=False, mutex=mutex, context=context)
        call_clbks(epoch_single_clbks, pop=True, mutex=mutex, context=context)


def kick():
    if TRAIN_FLAG == False:
        raise KeyboardInterrupt


def start_training():
    """
    Start training
    """
    global TRAIN_FLAG, TRAIN_THREAD
    if TRAIN_FLAG == False:
        TRAIN_FLAG = True
        TRAIN_THREAD = threading.Thread(
            target=train_fn,
            args=(
                BATCH_SINGLE_CLBKS,
                EPOCH_SINGLE_CLBKS,
                BATCH_PERSIST_CLBKS,
                EPOCH_PERSIST_CLBKS,
                kick,
                FILE,
                MUTEX,
            ),
        )
        print("Starting training", file=FILE, flush=True)
        TRAIN_THREAD.start()
    else:
        print("Already training, redirecting to view of output file")
        time.sleep(0.5)
    view_output_file()


@are_you_sure
def stop_training():
    """
    Stop training
    """
    global TRAIN_FLAG, TRAIN_THREAD
    if TRAIN_FLAG == True:
        TRAIN_FLAG = False
        TRAIN_THREAD.join()  # Exception tracebacks of thread will be shown at this stage
        print("Training stopped", file=FILE, flush=True)
        view_output_file()
    else:
        print("Not currently training anything")
        time.sleep(0.5)


def _import_function_from_module(modulename: str, funcname: str):
    try:
        module = importlib.import_module(modulename)
    except ModuleNotFoundError as e:
        print(f"Could not import module, got error: {e}")
        input()
        return None

    module = utils.reloader(module)  # Refreshes code if module loaded before

    try:
        func = module.__dict__[funcname]
    except KeyError as e:
        print(f'Could not find function "{funcname}" in module "{modulename}", got error {e}')
        input()
        return None
    return func


def delete_callback_in_queue(clbks: Queue):
    if clbks.empty():
        print("There are no callbacks here, press enter to return")
        input()
        return

    clbk: Callable
    for i, clbk in enumerate(clbks.queue):
        if doc := clbk.__doc__:
            doc = doc.strip().split("\n")[0]
        print(f"{i}. callable: {clbk}, doc: {doc}")

    try:
        choice = int(input())
    except ValueError as e:
        print("Invalid choice, must be integer")
        input()
        return

    if 0 < choice > i:
        print("Invalid choice, not within valid range")
        input()
        return

    del clbks.queue[choice]
    print("Deleted callback")
    input()


def insert_callback_in_queue(clbks: Queue, modulename: str, funcname: str):
    func = _import_function_from_module(modulename, funcname)
    if func is None:
        return

    clbks.put(func)
    input("Waiting for callback prints, press enter to return to menu\n\n")


def callback_menu(
    batch_single_clbks: Queue,
    epoch_single_clbks: Queue,
    batch_persist_clbks: Queue,
    epoch_persist_clbks: Queue,
):
    """
    Callback menu
    """

    def add_single_batch_callback(modulename: str, funcname: str):
        """
        Add \033[32msingle use\033[0m callback after \033[32mbatch\033[0m ends
        """
        insert_callback_in_queue(batch_single_clbks, modulename, funcname)

    def add_single_epoch_callback(modulename: str, funcname: str):
        """
        Add \033[32msingle use\033[0m callback after \033[32mepoch\033[0m ends
        """
        insert_callback_in_queue(epoch_single_clbks, modulename, funcname)

    def add_persistent_batch_callback(modulename: str, funcname: str):
        """
        Add \033[32mpersistent\033[0m callback after \033[32mbatch\033[0m ends
        """
        insert_callback_in_queue(batch_persist_clbks, modulename, funcname)

    def add_persistent_epoch_callback(modulename: str, funcname: str):
        """
        Add \033[32mpersistent\033[0m callback after \033[32mepoch\033[0m ends
        """
        insert_callback_in_queue(epoch_persist_clbks, modulename, funcname)

    def delete_persistent_batch_callback():
        """
        \033[31mDelete\033[0m \033[32mpersistent\033[0m callback after \033[32mbatch\033[0m ends
        """
        delete_callback_in_queue(batch_persist_clbks)

    def delete_persistent_epoch_callback():
        """
        \033[31mDelete\033[0m \033[32mpersistent\033[0m callback after \033[32mepoch\033[0m ends
        """
        delete_callback_in_queue(epoch_persist_clbks)

    menu(locals(), title=" Callback menu ")


@are_you_sure
def clear_output_file():
    """
    Clear output file
    """
    FILE.truncate(0)


def view_output_file():
    """
    View output file (live feed ASCII)
    """
    utils.monitor_file("output.txt", decode="ascii", mode="rb")


def view_status():
    """
    View status
    """
    utils.debug(TRAIN_FLAG)
    utils.debug(TRAIN_THREAD)
    BATCH_SINGLE_CALLBACK_QUEUE = list(BATCH_SINGLE_CLBKS.queue)
    EPOCH_SINGLE_CALLBACK_QUEUE = list(EPOCH_SINGLE_CLBKS.queue)
    BATCH_PERSIST_CALLBACK_QUEUE = list(BATCH_PERSIST_CLBKS.queue)
    EPOCH_PERSIST_CALLBACK_QUEUE = list(EPOCH_PERSIST_CLBKS.queue)
    utils.debug(BATCH_SINGLE_CALLBACK_QUEUE)
    utils.debug(EPOCH_SINGLE_CALLBACK_QUEUE)
    utils.debug(BATCH_PERSIST_CALLBACK_QUEUE)
    utils.debug(EPOCH_PERSIST_CALLBACK_QUEUE)

    print("Current runnig threads:")
    for t in threading.enumerate():
        print(t)

    input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlock", action="store_true")
    args = parser.parse_args()

    if args.unlock:
        TRAIN_FLAG: bool = False
        TRAIN_THREAD: Optional[threading.Thread] = None
        FILE: io.TextIOBase = open("output.txt", "a+")
        BYTEFILE: io.BufferedIOBase = open("output.txt", "rb")
        EPOCH_SINGLE_CLBKS: Queue = Queue()
        BATCH_SINGLE_CLBKS: Queue = Queue()
        BATCH_PERSIST_CLBKS: Queue = Queue()
        EPOCH_PERSIST_CLBKS: Queue = Queue()
        MUTEX = threading.Lock()
        cases = [
            start_training,
            stop_training,
            callback_menu,
            clear_output_file,
            view_status,
            view_output_file,
        ]
        case_kwargs = {
            callback_menu: {
                "batch_single_clbks": BATCH_SINGLE_CLBKS,
                "epoch_single_clbks": EPOCH_SINGLE_CLBKS,
                "batch_persist_clbks": BATCH_PERSIST_CLBKS,
                "epoch_persist_clbks": EPOCH_PERSIST_CLBKS,
            }
        }
        menu(cases, title=" Train manager ", main=True, case_kwargs=case_kwargs)
    else:
        cases = [
            view_output_file,
        ]
        menu(cases, title=" Train manager ", main=True)

    TRAIN_FLAG = False
    # TRAIN_THREAD.join()
