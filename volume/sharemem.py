import torch

def same_storage(x: torch.Tensor, y: torch.Tensor):
    xmin, xmax = x[0].data_ptr(), x[...,-1].data_ptr()
    ymin, ymax = y[0].data_ptr(), x[...,-1].data_ptr()
    print(min(xmax, ymax) - max(xmin, ymin))

if __name__ == "__main__":
    x = torch.arange(10)
    y = x[1::2]
    z = y.clone()

    from utils import debug
    debug(x)
    debug(y)
    debug(z)

    same_storage(x, y)
    same_storage(y, x)
    same_storage(x, z)
    same_storage(y, z)
