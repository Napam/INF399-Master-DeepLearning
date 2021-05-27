from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
from fishdetr3d import get_random_input, plot_output, plot_labels
from torchvision import transforms

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
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dropout=0.05
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
    def __init__(self, 
        num_classes,
        hidden_dim: int = 256,
        merge_hidden_dim: int = 128,
        fc_width: int=1024,
        fc_extra_layers: int = 6
    ):
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

        self.linear_class = get_decoder_fc(hidden_dim, num_classes + 1, fc_width, fc_extra_layers) # 8 layers
        self.linear_boxes = get_decoder_fc(hidden_dim, 6+3*2, fc_width, fc_extra_layers) # 8 layers

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
        pred_boxes_intermediate = self.linear_boxes(h[:, 1, :, :]) 
        # Avoid inplace, cuz autograd anxiety

        pred_boxes = torch.cat([
            pred_boxes_intermediate[:,:,(0,  1,  2)].tanh(),
            pred_boxes_intermediate[:,:,(3,  4,  5)],
            pred_boxes_intermediate[:,:,(6,  7,  8)].sin(),
            pred_boxes_intermediate[:,:,(9, 10, 11)].cos(),
        ], dim=2)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}


def normalize_rotations(rots: Union[np.ndarray, float]):
    """
    Normalize rotations, R -> [0, 1]
    """
    # Normalize then modulo
    # Rotations will become between [0, 1], where 0 is zero radians
    # and 1 is 2 pi radians
    # Python modulo: -0.75 % 1 -> 0.25
    return (rots / (2 * np.pi)) % 1


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

    def __init__(
        self,
        hidden_dim: int = 256,
        freeze_encoder: bool = False,
        fc_width: int = 1024,
        fc_extra_layers: int = 6,
        pretrained_enc: bool=True
    ):
        """
        num_classes: int, should be number of classes WITHOUT "no object" class
        """
        super().__init__()

        self.encoder = Encoder(hidden_dim=hidden_dim, pretrained=pretrained_enc)
        self.decoder = Decoder(6, fc_width=fc_width, fc_extra_layers=fc_extra_layers)

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
        
        for tgt in y:
            boxes = tgt['boxes']
            raw_angles = boxes[:,6:] * 2 * np.pi
            boxes = torch.cat([boxes[:,:6], raw_angles.sin(), raw_angles.cos()], axis=1)
            tgt['boxes'] = boxes

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

        for tgt in y:
            boxes = tgt['boxes']
            raw_angles = boxes[:,6:] * 2 * np.pi
            boxes = torch.cat([boxes[:,:6], raw_angles.sin(), raw_angles.cos()], axis=1)
            tgt['boxes'] = boxes

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


def postprocess_to_df(imgnrs: Sequence[int], output: DETROutput, thresh: float = 0.2):
    '''
    To use with Blender reconstruction script
    '''
    conf_list, classes_list, boxes_list = postprocess(output, thresh, to_numpy=True)
    assert len(conf_list) ==len(classes_list) == len(boxes_list), "conf, classes and boxes length mismatch"
    
    confs = chain.from_iterable(conf_list)
    classes = chain.from_iterable(classes_list)
    boxes = chain.from_iterable(boxes_list)
    imgnrs_repeated = chain.from_iterable((
        (imgnr,)*len(classes_) for imgnr, classes_ in zip(imgnrs, classes_list))
    )
    
    dict_df = {'conf': confs,'class_': classes}
    for attr, vals in zip(['x', 'y', 'z', 'w', 'l', 'h', 'rx', 'ry', 'rz'], zip(*boxes)):
        dict_df[attr] = vals
    return pd.DataFrame(dict_df, index=pd.Index(imgnrs_repeated, name="imgnr"))


def postprocess(
    output: DETROutput, thresh: float = 0.2, to_numpy: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    logitss = output["pred_logits"]
    boxess = output["pred_boxes"]

    confs_list = []
    classes_list = []
    boxes_list = []
    for logits, boxes in zip(logitss, boxess):
        class_conf, class_preds, box_preds = postprocess_sample(logits, boxes, thresh)
        if to_numpy:
            confs_list.append(class_conf.cpu().numpy())
            classes_list.append(class_preds.cpu().numpy())
            boxes_list.append(box_preds.cpu().numpy())
        else:
            confs_list.append(class_conf)
            classes_list.append(class_preds)
            boxes_list.append(box_preds)
    return confs_list, classes_list, boxes_list


def postprocess_sample(
    logits: torch.Tensor, boxes: torch.Tensor, thresh: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Returns prediction confidence, classification, boxes
    '''
    confidences = logits.softmax(-1)[:, :-1].max(-1)[0]
    keepmask = confidences > thresh

    if any(keepmask) == False:
        return torch.Tensor(), torch.Tensor(), torch.Tensor()
    
    # sincos encoding to raw radian angles
    boxes = boxes[keepmask]
    boxes = torch.column_stack([boxes[:,:6], (torch.atan2(boxes[:,[6,7,8]], boxes[:,[9,10,11]]) / (2*np.pi)) % 1])
    return confidences[keepmask], logits[keepmask].argmax(-1), 


if __name__ == "__main__":
    torch.hub.set_dir("./torch_cache/")
    model = FishDETR()
    X = get_random_input(4)

    # with torch.no_grad():
    #     output = model(X)
    #     debugs(output['pred_logits'])
    #     debugs(output['pred_boxes'])
    #     confs, cls, bx = postprocess(output, 0.15)
    print(model.state_dict()['decoder.linear_boxes.14.bias'])

    # lol = set()
    # model.apply(lambda x: lol.add(x.__class__.__name__))
    # debug(lol)
    # model.half()
