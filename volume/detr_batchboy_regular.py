from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import utils
from utils import debug, debugs, debugt
from matplotlib import pyplot as plt

StereoImgs = Tuple[torch.Tensor, torch.Tensor]
DETROutput = Dict[str, torch.Tensor]


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        pretrained: bool = True,
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dropout=0
        )

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

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

    def forward(self, X):
        # propagate X through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(X)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

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
        # propagate through the transformer
        h = self.transformer(src, tgt.repeat(1, N, 1)).transpose(0, 1)
        # Now shape is (N, 100, 256)
        # want to return (N, 1, 100, 256) to get "standard shape" for convolution purposes
        return h


class Decoder(nn.Module):
    def __init__(self, num_classes, hidden_dim: int = 256):
        """
        num_classes: int, should be number of classes WITHOUT "no object" class
        """
        super().__init__()

        self.linear_class = nn.Linear(in_features=hidden_dim, out_features=num_classes + 1)
        self.linear_bbox = nn.Linear(in_features=hidden_dim, out_features=4)

    def forward(self, h: torch.Tensor):
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid()
        }


class DETR(nn.Module):
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

    def __init__(self, hidden_dim: int = 256):
        """
        num_classes: int, should be number of classes WITHOUT "no object" class
        """
        super().__init__()

        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(6)

    def forward(self, X: torch.Tensor):
        h = self.encoder(X)
        return self.decoder(h)

    def train_on_batch(
        self,
        X: torch.Tensor,
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
        X: torch.Tensor,
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
        X: torch.Tensor,
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
            imgs = X.permute((0, 2, 3, 1)).cpu()
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


def postprocess(
    output: DETROutput, thresh: float = 0.2
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    logitss = output["pred_logits"]
    boxess = output["pred_boxes"]

    classes_list = []
    boxes_list = []
    for logits, boxes in zip(logitss, boxess):
        class_preds, box_preds = postprocess_sample(logits, boxes, thresh)
        classes_list.append(class_preds)
        boxes_list.append(box_preds)
    return classes_list, boxes_list


def postprocess_sample(
    logits: torch.Tensor, boxes: torch.Tensor, thresh: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    keepmask = logits.softmax(-1)[:, :-1].max(-1)[0] > thresh
    if any(keepmask) == False:
        return torch.Tensor(), torch.Tensor()
    return logits[keepmask].argmax(-1), boxes[keepmask]


def preprocess(X: StereoImgs, y: DETROutput, device: torch.device) -> Tuple[StereoImgs, DETROutput]:
    return preprocess_images(X, device), preprocess_labels(y, device)


def preprocess_images(images: torch.Tensor, device: Optional[torch.device] = None) -> List[tuple]:
    """
    images: (N,C,H,W)
    """
    return images.to(device)


def preprocess_labels(labels: Iterable, device: Optional[torch.device] = None) -> List[dict]:
    return [{k: v.to(device) for k, v in t.items()} for t in labels]


def get_random_input(
    N: int = 1, C: int = 3, H: int = 416, W: int = 416, device: Optional[torch.device] = None
) -> StereoImgs:
    """To test feedforward
    N: N in (N,C,H,W)
    """
    return torch.randn((N, C, H, W), device=device)


def collate(batch):
    '''
    Turns stereo to single by using only left image
    '''
    Xs, ys = zip(*batch)
    # Xs: Tuple[torch.Tensor]
    Xs: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ys: Tuple[DETROutput]
    return torch.cat([X[0] for X in Xs]), ys


@torch.no_grad()
def plot_output(
    imgs: ArrayLike, output: DETROutput, enforce_cpu: bool = True, thresh: float = 0.2, **kwargs
):
    """
    imgs: batch of imgs
    """
    if enforce_cpu:
        try:
            imgs = imgs.cpu()
        except AttributeError:
            pass

        output = {
            "pred_logits": output["pred_logits"].cpu(),
            "pred_boxes": output["pred_boxes"].cpu(),
        }

    class_predss, box_predss = postprocess(output, thresh)
    for img, class_preds, box_preds in zip(imgs, class_predss, box_predss):
        utils.plot_bboxes(img=img, classes=class_preds, boxes=box_preds, **kwargs)


def plot_labels(imgs: ArrayLike, labels: List[DETROutput], enforce_cpu: bool = True, **kwargs):
    for img, dict_ in zip(imgs, labels):
        if enforce_cpu:
            boxes = dict_["boxes"].cpu()
            classes = dict_["labels"].cpu()
        utils.plot_bboxes(img=img, classes=classes, boxes=boxes, **kwargs)


if __name__ == "__main__":
    torch.hub.set_dir("./torch_cache/")
    model = DETR()
    X = get_random_input(4)
    debugs(X)

    with torch.no_grad():
        output = model(X)
        debugs(output['pred_logits'])
        debugs(output['pred_boxes'])
        postprocess(output)