from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd 
from itertools import chain
from numpy.typing import ArrayLike
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import utils
from utils import debug, debugs, debugt
from matplotlib import pyplot as plt
from torchvision import transforms

StereoImgs = Tuple[torch.Tensor, torch.Tensor]
DETROutput = Dict[str, torch.Tensor]

from fishdetr3d import Encoder, plot_labels, plot_output, get_random_input


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()

        # No bias needed since batch norm
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channels)

        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=hidden_channels)

        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor):
        h1 = self.bn1(self.relu(self.conv1(h)))
        h2 = self.bn2(self.relu(self.conv2(h1)))
        h3 = self.bn3(self.relu(self.conv3(h2 + h1)))  # Skip connection
        return h3
        
def get_decoder_fc(in_features: int, out_features: int, width: int=256, extra_layers: int=3) -> torch.Tensor:
    # n_layers = 2 + extra_layers
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=width), nn.ReLU(inplace=True),
        *chain.from_iterable((nn.Linear(width, width), nn.ReLU(inplace=True)) for i in range(extra_layers)),
        nn.Linear(in_features=width, out_features=out_features),
    )

class Decoder(nn.Module):
    def __init__(self, num_classes, hidden_dim: int = 256, merge_hidden_dim: int = 128):
        """
        num_classes: int, should be number of classes WITHOUT "no object" class
        """
        super().__init__()

        self.block1 = DecoderBlock(
            in_channels=2, hidden_channels=merge_hidden_dim, out_channels=merge_hidden_dim
        )
        self.block2 = DecoderBlock(
            in_channels=merge_hidden_dim,
            hidden_channels=merge_hidden_dim,
            out_channels=merge_hidden_dim,
        )
        self.block3 = DecoderBlock(
            in_channels=merge_hidden_dim,
            hidden_channels=merge_hidden_dim,
            out_channels=merge_hidden_dim,
        )
        self.block4 = DecoderBlock(
            in_channels=merge_hidden_dim, hidden_channels=merge_hidden_dim, out_channels=2
        )

        self.linear_class = get_decoder_fc(hidden_dim, num_classes + 1, 1024, 6) # 8 layers
        self.linear_common = get_decoder_fc(hidden_dim, 1024, 1024, 4) # 6 layers
        self.linear_locs = get_decoder_fc(1024, 3, 341, 0) # 2 Layers
        self.linear_dims = get_decoder_fc(1024, 3, 341, 0) # 2 Layers
        self.linear_rots = get_decoder_fc(1024, 3, 341, 0) # 2 Layers

    def merge(self, h_left, h_right):
        # h_left and h_right: (N, 1, 100, 256)
        h = torch.cat((h_left, h_right), dim=1)  # (N, 2, 100, 256)

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)

        # (N, 2, 100, 256)
        return h

    def forward(self, h_left: torch.Tensor, h_right: torch.Tensor):
        """
        h_left: N, C, H, W
        h_right: N, C, H, W
        """
        h = self.merge(h_left, h_right).squeeze(1) # (N, 2, n_query, hidden_dim (256))
    
        # Each channel for logits and boxes
        pred_logits = self.linear_class(h[:, 0, :, :])
        # pred_boxes_intermediate = self.linear_boxes(h[:, 1, :, :]) 
        pred_common = self.linear_common(h[:, 1, :, :])
        pred_locs = self.linear_locs(pred_common)
        pred_dims = self.linear_dims(pred_common)
        pred_rots = self.linear_locs(pred_common)

        # Avoid inplace, cuz autograd anxiety
        pred_boxes = torch.cat([
            pred_locs.tanh(),
            pred_dims,
            pred_rots.sigmoid()
        ], dim=2)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}


class FishDETR(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(self, hidden_dim: int = 256, freeze_encoder: bool = False):
        """
        num_classes: int, should be number of classes WITHOUT "no object" class
        """
        super().__init__()

        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(6)

        if freeze_encoder:
            self.freeze_module(self.encoder)
            print("Encoder layers are frozen")

        self.freeze_encoder = freeze_encoder
        self.transform_imgs = transforms.Compose([
            transforms.Normalize([0.65629897,0.76457309,0.43896555], [0.06472352, 0.07107777, 0.05759248])
        ])

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, imgs: Tuple[torch.Tensor, torch.Tensor], *, callback: Optional[Callable] = None):
        # imgs[0] and imgs[1] should be (N, C, H, W)
        assert isinstance(imgs, (tuple, list))
        assert len(imgs) == 2

        h_left = self.encoder(self.transform_imgs(imgs[0]))
        h_right = self.encoder(self.transform_imgs(imgs[1]))

        output = self.decoder(h_left, h_right)
        
        if callback is not None:
            callback()
            
        return output

    def train_on_batch(
        self,
        X: StereoImgs,
        y: DETROutput,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        enforce_train: bool = False,
    ) -> Tuple[DETROutput, torch.Tensor]:
        """
        Train model on given batch of samples

        Will:
            1. forward pass
            2. calc loss
            3. zero gradient accumulation
            4. calc gradients
            5. update weights

        criterion: SetCriterion
        optimizer: torch.optim.Optimizer

        returns loss (float)
        """
        if enforce_train:
            self.train()
            criterion.train()

        output = self(X)
        loss_dict = criterion(output, y)
        weight_dict = criterion.weight_dict
        losses: torch.Tensor
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()  # Computes gradients
        optimizer.step()  # Do a gradient step
        return output, losses

    @torch.no_grad()
    def eval_on_batch(
        self,
        X: StereoImgs,
        y: DETROutput,
        criterion: nn.Module,
        enforce_eval: bool = False,
    ) -> Tuple[DETROutput, float]:
        if enforce_eval:
            self.eval()
            criterion.eval()

        output = self(X)
        loss_dict = criterion(output, y)
        weight_dict = criterion.weight_dict
        losses: torch.Tensor
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return output, losses

    @torch.no_grad()
    def forward_and_plot(
        self,
        X: StereoImgs,
        y: Optional[DETROutput] = None,
        output: Optional[DETROutput] = None,
        alt_imgs: Optional[ArrayLike] = None,
        thresh: float = 0.2,
        **kwargs
    ) -> None:

        self.eval()
        if output is None:
            output = self(X)

        # inp consists of left images
        if alt_imgs is None:
            # (N,C,H,W) -> (N,H,W,C) -> cpu
            imgs = X[0].permute((0, 2, 3, 1)).cpu()
        else:
            imgs = alt_imgs

        if y is not None:
            for img, logits, boxes, y_ in zip(imgs, output["pred_logits"], output["pred_boxes"], y):
                fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=kwargs.get("figsize", None))
                # Dummy stuff
                output_ = {"pred_logits": logits[None], "pred_boxes": boxes[None]}
                plot_output(img[None], output_, ax=ax_left, thresh=thresh, **kwargs)
                plot_labels(img[None, ...], [y_], ax=ax_right, **kwargs)
        else:
            plot_output(imgs, output, **kwargs)


if __name__ == "__main__":
    torch.hub.set_dir("../torch_cache/")
    model = FishDETR()
    X = get_random_input(4)

    with torch.no_grad():
        output = model(X)
        debugs(output['pred_logits'])
        debugs(output['pred_boxes'])
        # cls, bx = postprocess(output, 0.15)
        

    # lol = set()
    # model.apply(lambda x: lol.add(x.__class__.__name__))
    # debug(lol)
    # model.half()
