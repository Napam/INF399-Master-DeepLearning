from typing import Dict, Generic, Iterable, List, Optional, Tuple

import numpy as np
from torch.nn.modules.activation import ReLU
from torch.types import Number
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import utils
from utils import debug, debugs, debugt


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
        return h.unsqueeze(1)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # common kwargs
        kwargs = {"kernel_size": 1, "stride": 1}

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
        h3 = self.bn3(self.relu(self.conv3(h2 + h1)))
        return h3


class Decoder(nn.Module):
    def __init__(self, num_classes, hidden_dim: int = 256, merge_hidden_dim: int = 64):
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

        self.linear_pre_class = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear_pre_bbox = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        self.linear_class = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim * 2, out_features=num_classes + 1),
        )

        self.linear_bbox = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim * 2, out_features=4),
            nn.Sigmoid(),
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        # self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # self.linear_bbox = nn.Linear(hidden_dim, 4)

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
        # Output is (N, 2, n_query, n_classes)
        h = self.merge(h_left, h_right).squeeze(1)

        # finally project transformer outputs to class labels and bounding boxes
        return {
            # Each channel for logits and boxes
            "pred_logits": self.linear_class(h[:, 0, :, :]),
            "pred_boxes": self.linear_bbox(h[:, 1, :, :]),
        }


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

    def __init__(self, hidden_dim: int = 256, freeze_encoder: bool = True):
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

    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, imgs: Tuple[torch.Tensor, torch.Tensor]):
        # (1, 100, 256)
        # (batchsize, n_queries, embedding_dim)
        # imgs[0] and imgs[1] should be (N, C, H, W)
        assert isinstance(imgs, (tuple, list))
        assert len(imgs) == 2

        if self.freeze_encoder:
            self.encoder.eval()

        h_left = self.encoder(imgs[0])
        h_right = self.encoder(imgs[1])

        return self.decoder(h_left, h_right)

    def train_on_batch(
        self,
        X: Tuple[torch.Tensor, torch.Tensor],
        y: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        enforce_train: bool = False,
    ):
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
        X: Tuple[torch.Tensor, torch.Tensor],
        y: torch.Tensor,
        criterion: nn.Module,
        enforce_eval: bool = False,
    ):
        if enforce_eval:
            self.eval()
            criterion.eval()

        output = self(X)
        loss_dict = criterion(output, y)
        weight_dict = criterion.weight_dict
        losses: torch.Tensor
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return output, losses


def postprocess(logits: torch.Tensor, boxes: torch.Tensor, thresh: float = 0.2):
    keepmask = logits.softmax(-1)[:, :-1].max(-1)[0] > thresh
    if any(keepmask) == False:
        return torch.Tensor(), torch.Tensor()
    return logits[keepmask].argmax(-1), boxes[keepmask]


def img_handler(
    images: Tuple[torch.Tensor, torch.Tensor], device: Optional[torch.device] = None
) -> List[tuple]:
    '''
    Each tensor in Tuple[torch.Tensor, torch.Tensor] is (N,C,H,W)
    '''
    return (images[0].to(device), images[1].to(device))


def label_handler(labels: Iterable, device: Optional[torch.device] = None) -> List[dict]:
    return [{k: v.to(device) for k, v in t.items()} for t in labels]


def get_random_input(
    N: int = 1, C: int = 3, H: int = 800, W: int = 800, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """To test feedforward
    N: N in (N,C,H,W)
    """
    return (torch.randn((N, C, H, W), device=device), torch.randn((N, C, H, W), device=device))


def collate(batch):
    Xs, ys = zip(*batch)

    Xs: Tuple[Tuple[torch.Tensor]]
    ys: Tuple[Dict[str, torch.Tensor]]

    lefts = [None] * len(Xs)
    rights = [None] * len(Xs)
    for i, (left, right) in enumerate(Xs):
        lefts[i] = left
        rights[i] = right

    lefts = torch.cat(lefts)
    rights = torch.cat(rights)
    return [lefts, rights], ys


if __name__ == "__main__":
    torch.hub.set_dir("./torch_cache/")
    model = FishDETR()
    # X = get_random_input(2)
    # model(X)
    lol = set()
    model.apply(lambda x: lol.add(x.__class__.__name__))
    debug(lol)
    model.half()