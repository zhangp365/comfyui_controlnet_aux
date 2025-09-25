from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs
import comfy.model_management as model_management
from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
import time
import logging
logger = logging.getLogger(__file__)
class Depth_Anything_V2_Preprocessor:
    loaded_models = {}

    def __init__(self) -> None:
        self.ckpt_name = None

    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            ckpt_name=INPUT.COMBO(
                ["depth_anything_v2_vitg.pth", "depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"],
                default="depth_anything_v2_vitl.pth"
            ),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, ckpt_name, resolution=512, **kwargs):
        logger.info(f"Depth Anything V2 Preprocessor execute, ckpt_name: {ckpt_name}")
        start_time = time.time()
        if ckpt_name != self.ckpt_name:
            self.ckpt_name = ckpt_name
            if ckpt_name not in self.loaded_models:
                self.loaded_models[ckpt_name] = DepthAnythingV2Detector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
        
        model = self.loaded_models[ckpt_name]
        logger.info(f"Depth Anything V2 Preprocessor model loaded cost: {time.time() - start_time}")
        out = common_annotator_call(model, image, resolution=resolution, max_depth=1)
        logger.info(f"Depth Anything V2 Preprocessor execute, cost: {time.time() - start_time}")
        return (out, )

""" class Depth_Anything_Metric_V2_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            environment=(["indoor", "outdoor"], {"default": "indoor"}),
            max_depth=("FLOAT", {"min": 0, "max": 100, "default": 20.0, "step": 0.01})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, environment, resolution=512, max_depth=20.0, **kwargs):
        from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
        filename = dict(indoor="depth_anything_v2_metric_hypersim_vitl.pth", outdoor="depth_anything_v2_metric_vkitti_vitl.pth")[environment]
        model = DepthAnythingV2Detector.from_pretrained(filename=filename).to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, max_depth=max_depth)
        del model
        return (out, ) """

NODE_CLASS_MAPPINGS = {
    "DepthAnythingV2Preprocessor": Depth_Anything_V2_Preprocessor,
    #"Metric_DepthAnythingV2Preprocessor": Depth_Anything_Metric_V2_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnythingV2Preprocessor": "Depth Anything V2 - Relative",
    #"Metric_DepthAnythingV2Preprocessor": "Depth Anything V2 - Metric"
}