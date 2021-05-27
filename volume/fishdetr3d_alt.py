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

from fishdetr3d import plot_labels, plot_output, get_random_input, DecoderBlock, postprocess_to_df

StereoImgs = Tuple[torch.Tensor, torch.Tensor]
DETROutput = Dict[str, torch.Tensor]


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        pretrained: bool = True,
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth",
            map_location="cpu",
            check_hash=True,
        )

        temp_state_dict = self.state_dict()
        utils.dict_union_update(temp_state_dict, state_dict)
        self.load_state_dict(temp_state_dict)
        print("Encoder successfully loaded with pretrained weights")

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        return self.conv(x)


def get_decoder_fc(
    in_features: int, out_features: int, width: int = 256, extra_layers: int = 3
) -> torch.Tensor:
    # n_layers = 2 + extra_layers
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=width),
        nn.ReLU(inplace=True),
        *chain.from_iterable(
            (nn.Linear(width, width), nn.ReLU(inplace=True)) for i in range(extra_layers)
        ),
        nn.Linear(in_features=width, out_features=out_features),
    )


class Decoder(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim: int = 256,
        nheads: int = 16,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        """
        num_classes: int, should be number of classes WITHOUT "no object" class
        """
        super().__init__()

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim*2, nheads, num_encoder_layers, num_decoder_layers, dropout=0.05
        )

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim*2))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim))

        self.linear_class = get_decoder_fc(hidden_dim*2, num_classes + 1, 1024, 6)  # 8 layers
        self.linear_boxes = get_decoder_fc(hidden_dim*2, 9, 1024, 6)  # 8 layers
    

    def forward(self, h_left: torch.Tensor, h_right: torch.Tensor):
        """
        h_left: N, C, H, W
        h_right: N, C, H, W
        """
        h = torch.cat([h_left, h_right], axis=1)

        # construct positional encodings
        H, W = h.shape[-2:]

        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        src = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        tgt = self.query_pos.unsqueeze(1)

        N = src.shape[1]
        h = self.transformer(src, tgt.repeat(1, N, 1)).transpose(0, 1)

        # Each channel for logits and boxes
        pred_logits = self.linear_class(h)
        pred_boxes_intermediate = self.linear_boxes(h)
        # Avoid inplace, cuz autograd anxiety
        pred_boxes = torch.cat(
            [
                pred_boxes_intermediate[:, :, (0, 1, 2)].tanh(),
                pred_boxes_intermediate[:, :, (3, 4, 5)],
                pred_boxes_intermediate[:, :, (6, 7, 8)].sigmoid(),
            ],
            dim=2,
        )

        # finally project transformer outputs to class labels and bounding boxes
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


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

    def __init__(self, hidden_dim: int = 256, freeze_encoder: bool = False, pretrained_enc: bool=True):
        """
        num_classes: int, should be number of classes WITHOUT "no object" class
        """
        super().__init__()

        self.encoder = Encoder(hidden_dim=hidden_dim, pretrained=pretrained_enc)
        self.decoder = Decoder(6)

        if freeze_encoder:
            self.freeze_module(self.encoder)
            print("Encoder layers are frozen")

        self.freeze_encoder = freeze_encoder
        self.transform_imgs = transforms.Compose(
            [
                transforms.Normalize(
                    [0.65629897, 0.76457309, 0.43896555], [0.06472352, 0.07107777, 0.05759248]
                )
            ]
        )

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(
        self, imgs: Tuple[torch.Tensor, torch.Tensor], *, callback: Optional[Callable] = None
    ):
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
        debugs(output["pred_logits"])
        debugs(output["pred_boxes"])
        # cls, bx = postprocess(output, 0.15)

    # lol = set()
    # model.apply(lambda x: lol.add(x.__class__.__name__))
    # debug(lol)
    # model.half()
