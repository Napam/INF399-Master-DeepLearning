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


def call_callbacks(
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
    batch_single_callbacks: Queue,
    epoch_single_callbacks: Queue,
    batch_persist_callbacks: Queue,
    epoch_persist_callbacks: Queue,
    stop_callback: Callable,
    file: io.BufferedIOBase,
    mutex: threading.Lock,
):
    context = {
        "n_epochs": 1000,
        "n_batches": None,
        "epoch": None,
        "batch": None,
        "epoch_single_callbacks": epoch_single_callbacks,
        "batch_single_callbacks": batch_single_callbacks,
        "batch_persist_callbacks": batch_persist_callbacks,
        "epoch_persist_callbacks": epoch_persist_callbacks,
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
            call_callbacks(batch_persist_callbacks, pop=False, mutex=mutex, context=context)
            call_callbacks(batch_single_callbacks, pop=True, mutex=mutex, context=context)
        call_callbacks(epoch_persist_callbacks, pop=False, mutex=mutex, context=context)
        call_callbacks(epoch_single_callbacks, pop=True, mutex=mutex, context=context)


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
            target=train_fn, args=(BATCH_SINGLE_CLBKS, EPOCH_SINGLE_CLBKS, kick, FILE, MUTEX)
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
        print(f"Could import module, got {e}")

    module = utils.reloader(module)  # Refreshes code if module loaded before

    try:
        func = module.__dict__[funcname]
    except KeyError as e:
        print(f'Could not find function "{funcname}" in module "{modulename}", got error {e}')
    return func


def callback_menu():
    """
    Callback menu
    """

    def add_single_batch_callback(modulename: str, funcname: str):
        """
        Add \033[32msingle use\033[0m callback after \033[32mbatch\033[0m ends
        """
        func = _import_function_from_module(modulename, funcname)
        BATCH_SINGLE_CLBKS.put(func)
        input("Waiting for callback prints, press enter to return to menu\n\n")

    def add_single_epoch_callback(modulename: str, funcname: str):
        """
        Add \033[32msingle use\033[0m callback after \033[32mepoch\033[0m ends
        """
        func = _import_function_from_module(modulename, funcname)
        EPOCH_SINGLE_CLBKS.put(func)
        input("Waiting for callback prints, press enter to return to menu\n\n")

    def add_persistent_batch_callback(modulename: str, funcname: str):
        """
        Add \033[32mpersistent\033[0m callback after \033[32mbatch\033[0m ends
        """
        func = _import_function_from_module(modulename, funcname)
        BATCH_PERSIST_CLBKS.put(func)
        input("Waiting for callback prints, press enter to return to menu\n\n")

    def add_persistent_epoch_callback(modulename: str, funcname: str):
        """
        Add \033[32mpersistent\033[0m callback after \033[32mepoch\033[0m ends
        """
        func = _import_function_from_module(modulename, funcname)
        EPOCH_PERSIST_CLBKS.put(func)
        input("Waiting for callback prints, press enter to return to menu\n\n")

    def delete_persistent_batch_callback():
        """
        \033[31mDelete\033[0m \033[32msingle use\033[0m callback after \033[32mepoch\033[0m ends
        """

    def delete_persistent_epoch_callback():
        """
        \033[31mDelete\033[0m \033[32mpersistent\033[0m callback after \033[32mepoch\033[0m ends
        """

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

    BATCH_CALLBACK_QUEUE = list(BATCH_SINGLE_CLBKS.queue)
    EPOCH_CALLBACK_QUEUE = list(EPOCH_SINGLE_CLBKS.queue)
    utils.debug(BATCH_CALLBACK_QUEUE)
    utils.debug(EPOCH_CALLBACK_QUEUE)

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
        BATCH_ALL_CLBKS: deque = deque()
        EPOCH_ALL_CLBKS: deque = deque()
        MUTEX = threading.Lock()
        cases = [
            start_training,
            stop_training,
            callback_menu,
            clear_output_file,
            view_status,
            view_output_file,
        ]
    else:
        cases = [
            view_output_file,
        ]

    menu(cases, title=" Train manager ", blank_proceedure="pass", main=True)
