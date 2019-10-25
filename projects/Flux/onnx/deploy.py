from typing import List, Tuple, Optional
from pathlib import Path

import torch as pt
from torch import nn
from torch.onnx.symbolic_helper import parse_args

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import Backbone, build_model
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


def retinanet2onnx(
        model_path: Path,
        cfg_path: Path,
        onnx_path: Path,
        input_shape: Tuple[int],
        input_names: List[str], 
        output_names: List[str], 
        opset: int
):
    r"""
    Export a RetinaNet model as an ONNX file.

    Paramters
    ---------
    model_path
    cfg_path
    onnx_path
    input_shape
    input_names
    output_names
    opset

    """

    # Load configurations.
    cfg = get_cfg()
    cfg.merge_from_file(config_path)

    # Build and load a RetinaNet model.
    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(model_path)

    # Create a dummy input tensor (batch, channels, height, width)
    if len(input_shape) != 4: 
        raise ValueError(f'input_shape must be (B, C, H, W)')
    dummy_input = pt.randn(input_shape).to(model.device)

    # Export ONNX.
    model.export(
        onnx_path,
        dummy_input,
        input_names,
        output_names,
        opset
    )


if __name__ == '__main__':
    from fire import Fire
    Fire(retinanet2onnx)
