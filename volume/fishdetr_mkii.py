from typing import Generic, Optional, Tuple

from typing_extensions import IntVar
import numpy as np
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import utils
from utils import debugt
    

class Encoder(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, pretrained: bool=True):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        
        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dropout=0)

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
            url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
            map_location='cpu',
            check_hash=True
        )
    
        self_state_dict = self.state_dict()
        utils.dict_union_update(self_state_dict, state_dict)
        self.load_state_dict(self_state_dict)
        print('Encoder successfully loaded with pretrained weights')

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
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

    
        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        # Now shape is (1, 100, 256)
        
        # want to return (1, 1, 100, 256) to get "standard shape"
        return h.unsqueeze(0)
    
    
class Decoder(nn.Module):
    def __init__(self, num_classes, hidden_dim: int=256, merge_hidden_dim: int = 64):
        '''
        num_classes: int, should be number of classes WITHOUT "no object" class
        '''
        super().__init__()
            
        # common kwargs
        kwargs = {'kernel_size':1, 'stride':1}

        # For latent space merging, use 1x1 convs
        self.merger1 = nn.Conv2d(in_channels=2, out_channels=merge_hidden_dim, **kwargs)
        self.merger2 = nn.Conv2d(in_channels=merge_hidden_dim, out_channels=merge_hidden_dim, **kwargs)
        self.merger3 = nn.Conv2d(in_channels=merge_hidden_dim, out_channels=1, **kwargs)
        
        self.linear_pre_class = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear_pre_bbox = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        
        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
    
    
    def merge(self, h_left, h_right):
        # h_left and h_right: (N, 1, 100, 256)
        h1 = torch.cat((h_left, h_right), dim=1) # 2 channel out
        
        h1 = self.merger1(h1) # 64 channel out
        h1 = F.relu(h1)
        h2 = self.merger2(h1) # 64 channel out
        h2 = F.relu(h1+h2)  # Skip connection
        h2 = self.merger3(h2) # 1 channel out
        h2 = F.relu(h2)
        
        # (1, 100, 256)
        return h2
        
    def forward(self, h_left: torch.Tensor, h_right: torch.Tensor):
        '''
        h_left: N, C, H, W
        h_right: N, C, H, W
        '''
        # Output is (N, 1, n_query, n_classes)
        h = self.merge(h_left, h_right).squeeze(1)
        
        h_logits = F.relu(self.linear_pre_class(h))
        h_boxes = F.relu(self.linear_pre_bbox(h))
    
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h_logits),
                'pred_boxes': self.linear_bbox(h_boxes).sigmoid()}
    

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
    def __init__(self, hidden_dim: int=256, freeze_encoder: bool=True):
        '''
        num_classes: int, should be number of classes WITHOUT "no object" class
        '''
        super().__init__()

        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(6)
        
        if freeze_encoder:
            self.freeze_module(self.encoder)
            print('Encoder layers are frozen')
            
        self.freeze_encoder = freeze_encoder
        
    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
                param.requires_grad = False
        
    def forward(self, imgs: Tuple[torch.Tensor, torch.Tensor]):
        # (1, 100, 256)
        # (batchsize, n_queries, embedding_dim)
        # imgs[0] and imgs[1] should be (N, C, H, W)    
        self.encoder.eval()

        h_left = self.encoder(imgs[0])
        h_right = self.encoder(imgs[1])
    
        return self.decoder(h_left, h_right)

def get_random_input(N: int=1, device: Optional[torch.device]=None):
    '''To test feedforward
    N: N in (N,C,H,W)
    '''
    return (
        torch.randn((N, 3, 800, 800), device=device),
        torch.randn((N, 3, 800, 800), device=device)
    )