from typing import Any, Callable, Optional
from pypatconsole import menu
import threading
import time
import queue
import utils
import io
from functools import wraps
import tqdm
import argparse
# import train_fn
import train_fn as tf
import sys
import multiprocessing

def are_you_sure(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f"Are you sure you want to execute {f.__name__}? y/n")
        inp = input()
        if inp == "y":
            output = f(*args, **kwargs)
            return output
    return wrapper


# def train_fn(batch_queue: queue.Queue, epoch_queue: queue.Queue, stop_callback: Callable, file):
#     i = 0
#     while i := i + 1:
#         if i > 2:
#             break
#         args = (10,)
#         kwargs = dict(file=file, ascii=True, desc="Test")

#         # if not batch_queue.empty():
#             # args, kwargs = epoch_queue.get()(*args, **kwargs)

#         for j in tqdm.trange(*args, **kwargs):
#             stop_callback()
#             time.sleep(0.5)


def train_fn_outer(*args):
    t = threading.Thread(target=tf.train_fn, args=args)
    t.start()
    t.join()
    file = args[-1]
    print("Process finished successfully", file=file, flush=True)
    global TRAIN_FLAG 
    TRAIN_FLAG = False
    # del sys.modules["tf"]
    # del tf
    del threading.enumerate()[-1]
    return


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
        TRAIN_THREAD = threading.Thread(target=train_fn_outer, args=(BATCH_CLBK_QUEUE, EPOCH_CLBK_QUEUE, kick, FILE))
        print('Starting training', file=FILE, flush=True)
        TRAIN_THREAD.start()
        view_output_file()
    else:
        print('Already training, redirecting to view of output file')
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
        TRAIN_THREAD.join() # Exception tracebacks of thread will be shown at this stage
        print("Training stopped", file=FILE, flush=True)
        view_output_file()
    else:
        print("Not currently training anything")
        time.sleep(0.5)


@are_you_sure
def add_batch_callback():
    """
    Add callback after batch ends
    """
    pass


@are_you_sure
def add_epoch_callback():
    """
    Add callback after epoch ends
    """
    pass


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
    '''
    View status
    '''
    utils.debug(TRAIN_FLAG)
    utils.debug(TRAIN_THREAD)

    BATCH_CALLBACK_QUEUE = list(BATCH_CLBK_QUEUE.queue)
    EPOCH_CALLBACK_QUEUE = list(BATCH_CLBK_QUEUE.queue)
    utils.debug(BATCH_CALLBACK_QUEUE)
    utils.debug(EPOCH_CALLBACK_QUEUE)

    print('Current runnig threads:')
    for t in threading.enumerate():
        print(t)
    
    input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--unlock', action='store_true')
    args = parser.parse_args()

    if args.unlock:
        TRAIN_FLAG: bool = False
        TRAIN_THREAD: Optional[threading.Thread] = None
        FILE: io.TextIOBase = open("output.txt", "a+")
        BYTEFILE: io.BufferedIOBase = open("output.txt", "rb")
        BATCH_CLBK_QUEUE: queue.Queue = queue.Queue()
        EPOCH_CLBK_QUEUE: queue.Queue = queue.Queue()
        cases = [
            start_training,
            stop_training,
            add_batch_callback,
            add_epoch_callback,
            clear_output_file,
            view_status,
            view_output_file,
        ]
    else:
        cases = [
            view_output_file,
        ]

    menu(cases, title=" Train manager ", blank_proceedure="pass")