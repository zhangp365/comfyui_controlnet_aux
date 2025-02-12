from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management
import numpy as np
import warnings
from custom_controlnet_aux.dwpose import DwposeDetector, AnimalposeDetector
from custom_controlnet_aux.dwpose.util import draw_facepose
import os
import json
import cv2
import math
import torch
from math import dist

DWPOSE_MODEL_NAME = "yzd-v/DWPose"
#Trigger startup caching for onnxruntime
GPU_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider"]
def check_ort_gpu():
    try:
        import onnxruntime as ort
        for provider in GPU_PROVIDERS:
            if provider in ort.get_available_providers():
                return True
        return False
    except:
        return False

if not os.environ.get("DWPOSE_ONNXRT_CHECKED"):
    if check_ort_gpu():
        print("DWPose: Onnxruntime with acceleration providers detected")
    else:
        warnings.warn("DWPose: Onnxruntime not found or doesn't come with acceleration providers, switch to OpenCV with CPU device. DWPose might run very slowly")
        os.environ['AUX_ORT_PROVIDERS'] = ''
    os.environ["DWPOSE_ONNXRT_CHECKED"] = '1'


class Keypoint:
    def __init__(self, x: float, y: float, confidence: float = 1.0):
        self.x = x  # X-coordinate (normalized or pixel-based)
        self.y = y  # Y-coordinate (normalized or pixel-based)
        self.confidence = confidence  # Confidence score of the keypoint

class DWPose_Preprocessor:
    def __init__(self) -> None:
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            detect_hand=INPUT.COMBO(["enable", "disable"]),
            detect_body=INPUT.COMBO(["enable", "disable"]),
            detect_face=INPUT.COMBO(["enable", "disable"]),
            resolution=INPUT.RESOLUTION(),
            bbox_detector=INPUT.COMBO(
                ["yolox_l.torchscript.pt", "yolox_l.onnx", "yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx"],
                default="yolox_l.onnx"
            ),
            pose_estimator=INPUT.COMBO(
                ["dw-ll_ucoco_384_bs5.torchscript.pt", "dw-ll_ucoco_384.onnx", "dw-ll_ucoco.onnx"],
                default="dw-ll_ucoco_384_bs5.torchscript.pt"
            ),
            scale_stick_for_xinsr_cn=INPUT.COMBO(["disable", "enable"]),
            max_people_number=INPUT.INT()
        )

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("normal", "POSE_KEYPOINT","xl_pose_body_only", "b&w_pose")
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def estimate_pose(self, image, detect_hand="enable", detect_body="enable", detect_face="enable", resolution=512, bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx", scale_stick_for_xinsr_cn="disable", max_people_number=0, **kwargs):
        if bbox_detector == "yolox_l.onnx":
            yolo_repo = DWPOSE_MODEL_NAME
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        else:
            raise NotImplementedError(f"Download mechanism for {bbox_detector}")

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = DWPOSE_MODEL_NAME
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            raise NotImplementedError(f"Download mechanism for {pose_estimator}")

        if self.model is None:
            self.model = DwposeDetector.from_pretrained(
                pose_repo,
                yolo_repo,
                det_filename=bbox_detector, pose_filename=pose_estimator,
                torchscript_device=model_management.get_torch_device()
            )
        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"
        scale_stick_for_xinsr_cn = scale_stick_for_xinsr_cn == "enable"
        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = self.model(image, **kwargs)
            self.openpose_dicts.append(openpose_dict)
            return pose_img

        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution, xinsr_stick_scaling=scale_stick_for_xinsr_cn)
     
        people = self.openpose_dicts[0].get('people', [{}])

        people_keypoints = [self.convert_keypoints(one) for one in people]
        if len(people_keypoints) > max_people_number and max_people_number > 0:
            people_keypoints, indices = self.remove_unavaible_keypoints(people_keypoints, max_people_number)
            if len(indices) > 0 and len(indices) < len(people):
                self.openpose_dicts[0]["people"] = [people[i] for i in indices]
        
        canvas = np.zeros((self.openpose_dicts[0].get("canvas_height"),self.openpose_dicts[0].get("canvas_width"),3))
        for keypoints in people_keypoints:
            canvas = draw_bodypose(canvas, keypoints)
        
        canvas_bw = np.zeros((self.openpose_dicts[0].get("canvas_height"),self.openpose_dicts[0].get("canvas_width"),3))
        for keypoints in people_keypoints:
            canvas_bw = draw_bodypose(canvas_bw, keypoints, bw=True)
        
        if detect_face:
            face_keypoints =[self.convert_keypoints(one, key="face_keypoints_2d") for one in self.openpose_dicts[0]["people"]]
            face_keypoints = [point for ps in face_keypoints for point in ps]
            canvas_bw = draw_facepose(canvas_bw, face_keypoints)

        pose_image_new = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
        pose_image_bw = torch.from_numpy(canvas_bw.astype(np.float32) / 255.0).unsqueeze(0)
        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts, pose_image_new, pose_image_bw)
        }

    def remove_unavaible_keypoints(self, people_keypoints, max_people_number):
        max_lengths = []
        indices = list(range(len(people_keypoints)))  # 添加索引追踪

        # Calculate the maximum length for each group of keypoints
        for keypoints in people_keypoints:
            max_length = self.get_max_distance(keypoints)
            max_lengths.append(max_length)

        # 将索引也加入排序
        sorted_tuple = sorted(zip(people_keypoints, max_lengths, indices), 
                            key=lambda x: x[1], 
                            reverse=True)

        # 解压缩排序后的元组
        filtered_keypoints = []
        kept_indices = []
        
        for keypoints, _, idx in sorted_tuple[:max_people_number]:
            filtered_keypoints.append(keypoints)
            kept_indices.append(idx)

        return filtered_keypoints, kept_indices

    def get_max_distance(self, keypoints):
        keypoints = [p for p in keypoints if p.confidence!=0]
        max_distance = 0

        # Calculate the distance between every pair of keypoints
        for i in range(len(keypoints)):
            for j in range(i + 1, len(keypoints)):
                if keypoints[i] is not None and keypoints[j] is not None:
                    distance = dist((keypoints[i].x, keypoints[i].y), (keypoints[j].x, keypoints[j].y))
                    max_distance = max(max_distance, distance)

        return max_distance
    
    def convert_keypoints(self, one, key="pose_keypoints_2d"):
        pose_keypoints_2d = one.get(key,[])
        keypoints = []
        for i in range(len(pose_keypoints_2d)//3):
            keypoint = Keypoint(*pose_keypoints_2d[3*i:3*(i+1)])
            keypoints.append(keypoint)
        return keypoints



def draw_bodypose(canvas: np.ndarray, keypoints: list, bw=False) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    """
    H, W, C = canvas.shape

    
    if max(W, H) < 500:
        ratio = 1.0
    elif max(W, H) >= 500 and max(W, H) < 1000:
        ratio = 2.0
    elif max(W, H) >= 1000 and max(W, H) < 2000:
        ratio = 3.0
    elif max(W, H) >= 2000 and max(W, H) < 3000:
        ratio = 4.0
    elif max(W, H) >= 3000 and max(W, H) < 4000:
        ratio = 5.0
    elif max(W, H) >= 4000 and max(W, H) < 5000:
        ratio = 6.0
    else:
        ratio = 7.0

    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    if bw:
        colors = [[255, 255, 255] for _ in colors]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]
        if keypoint1 is None or keypoint2 is None or keypoint1.confidence == 0 or  keypoint2.confidence == 0:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) 
        X = np.array([keypoint1.y, keypoint2.y]) 
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), int(stickwidth * ratio)), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None or keypoint.confidence == 0:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x)
        y = int(y)
        cv2.circle(canvas, (int(x), int(y)), int(4 * ratio), color, thickness=-1)

    return canvas

class AnimalPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            bbox_detector = INPUT.COMBO(
                ["yolox_l.torchscript.pt", "yolox_l.onnx", "yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx"],
                default="yolox_l.torchscript.pt"
            ),
            pose_estimator = INPUT.COMBO(
                ["rtmpose-m_ap10k_256_bs5.torchscript.pt", "rtmpose-m_ap10k_256.onnx"],
                default="rtmpose-m_ap10k_256_bs5.torchscript.pt"
            ),
            resolution = INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def estimate_pose(self, image, resolution=512, bbox_detector="yolox_l.onnx", pose_estimator="rtmpose-m_ap10k_256.onnx", **kwargs):
        if bbox_detector == "yolox_l.onnx":
            yolo_repo = DWPOSE_MODEL_NAME
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        else:
            raise NotImplementedError(f"Download mechanism for {bbox_detector}")

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = DWPOSE_MODEL_NAME
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            raise NotImplementedError(f"Download mechanism for {pose_estimator}")

        model = AnimalposeDetector.from_pretrained(
            pose_repo,
            yolo_repo,
            det_filename=bbox_detector, pose_filename=pose_estimator,
            torchscript_device=model_management.get_torch_device()
        )

        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = model(image, **kwargs)
            self.openpose_dicts.append(openpose_dict)
            return pose_img

        out = common_annotator_call(func, image, image_and_json=True, resolution=resolution)
        del model
        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts)
        }

NODE_CLASS_MAPPINGS = {
    "DWPreprocessor": DWPose_Preprocessor,
    "AnimalPosePreprocessor": AnimalPose_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DWPreprocessor": "DWPose Estimator",
    "AnimalPosePreprocessor": "AnimalPose Estimator (AP10K)"
}