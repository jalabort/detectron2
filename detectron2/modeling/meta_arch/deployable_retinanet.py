import torch
import io
from torch import nn


class DeployableDetectron2Model(nn.Module):
    def __init__(self, detectron2_model):
        super().__init__()
        self.backbone = detectron2_model.backbone
        self.head = detectron2_model.head
        self.in_features = detectron2_model.in_features
        self.anchor_generator = detectron2_model.anchor_generator
        self.nms = detectron2_model.nms
        self.score_threshold = detectron2_model.score_threshold
        self.topk_candidates = detectron2_model.topk_candidates
        self.nms_threshold = detectron2_model.nms_threshold
        self.max_detections_per_image = detectron2_model.max_detections_per_image

        self.exporting = False
        
    def forward(self, x):
        features = self.backbone(x)
        features = [features[f] for f in self.in_features]
        return self.head(features)

    def export(self, size, batch, precision, verbose=True, onnx_only=False, opset=None):
        '''
        Partly copied from the exporting section of 
        https://github.com/NVIDIA/retinanet-examples/blob/master/retinanet/model.py
        '''
        # import torch.onnx.symbolic

        # if opset is not None and opset < 9:
        #     # Override Upsample's ONNX export from old opset if required (not needed for TRT 5.1+)
        #     @torch.onnx.symbolic.parse_args('v', 'is')
        #     def upsample_nearest2d(g, input, output_size):
        #         height_scale = float(output_size[-2]) / input.type().sizes()[-2]
        #         width_scale = float(output_size[-1]) / input.type().sizes()[-1]
        #         return g.op("Upsample", input,
        #             scales_f=(1, 1, height_scale, width_scale),
        #             mode_s="nearest")
        #     torch.onnx.symbolic.upsample_nearest2d = upsample_nearest2d

        # Export to ONNX
        print('Exporting to ONNX...')
        self.exporting = True
        onnx_bytes = io.BytesIO()
        zero_input = torch.randn(1, 3, *size).cuda()
        extra_args = { 'opset_version': opset } if opset else {}
        torch.onnx.export(self.cuda(), zero_input, onnx_bytes, *extra_args)

        if onnx_only:
            return onnx_bytes.getvalue()

        # Build TensorRT engine
        # model_name = '_'.join([k for k, _ in self.backbones.items()])
        model_name = 'test.plan'

        # This is not ideal but keep for now. Replace with TensorRT gridAnchorPlugin and run from withing engine
        features = self.backbone(zero_input)
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        # Convert from Box objects to their tensor equivalent
        anchors = [[anchors[i][j].tensor for j in range(len(anchors[i]))] for i in range(len(anchors))]

        engine = Engine(onnx_bytes.getvalue(), len(onnx_bytes.getvalue()), batch, precision, 
                        self.score_threshold, self.topk_candidates, anchors, 
                        self.nms_threshold, self.max_detections_per_image, model_name, verbose)

        return anchors