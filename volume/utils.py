import torch as th

def pytorch_init():
    th.cuda.set_device(1)
    th.cuda.current_device()
    
    # Sanity checks
    assert th.cuda.current_device() == 1, 'Using wrong GPU'
    assert th.cuda.device_count() == 2, 'Cannot find both GPUs'
    assert th.cuda.get_device_name(0) == 'GeForce RTX 2080 Ti'
    assert th.cuda.is_available() == True, 'GPU not available'
    
pytorch_init()