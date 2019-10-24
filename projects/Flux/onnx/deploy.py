from typing import List
from pathlib import Path

import torch as pt
from torch import nn
from torch.onnx.symbolic_helper import parse_args

from detectron2.modeling import Backbone
from detectron2.modeling.meta_arch.retinanet import (
    RetinaNet,
    RetinaNetHead
)


@parse_args('v', 'is')
def upsample_nearest2d(g, x, output_size):
    h = float(output_size[-2]) / x.type().sizes()[-2]
    w = float(output_size[-1]) / x.type().sizes()[-1]
    return g.op(
        'Upsample', 
        x,
        scales_f=(1, 1, h, w),
        mode_s='nearest'
    )


class DeployableRetinaNet(nn.Module):
    def __init__(
            self, 
            model: RetinaNet
    ):
        super().__init__()
        self.backbone: Backbone = model.backbone
        self.head: RetinaNetHead = model.head
        self.in_features = model.in_features
        
    def forward(
            self, 
            x: pt.Tensor
    ):
        features = self.backbone(x)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        return box_cls, box_delta
    
    def export(
            self, 
            filename: Path, 
            dummy_input: pt.Tensor, 
            input_names: List[str], 
            output_names: List[str], 
            opset: int = 10
    ):
        # NOTE @jinyeom:
        #   This ensures that the model can be deployed to TensorRT with opset < 9.
        #   https://github.com/NVIDIA/retinanet-examples/blob/master/retinanet/model.py#L218
        if opset < 9:
            pt.onnx.symbolic_helper.upsample_nearest2d = upsample_nearest2d

        pt.onnx.export(
            self, 
            dummy_input, 
            filename,
            export_params=True,
            opset_version=opset,
            input_names=input_names, 
            output_names=output_names
        )