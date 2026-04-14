import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

try:
    import torch
    print(f":::::TORCH VERSION: {torch.__version__}")
    _ = torch.empty(1).cuda() 
    print(":::::TORCH IS READY:::::")
except Exception as e:
    print(f":::::TORCH FAILED: {e}:::::")

try:
    import paddle
    paddle.set_device('gpu')
    from paddleocr import PaddleOCR
    print(":::::PADDLE IS READY:::::")
    
    ocr = PaddleOCR(use_textline_orientation=True, lang='en', show_log=False)
    print(":::::ALL SYSTEMS ARE READY:::::")
except Exception as e:
    print(f":::::PADDLE FAILED: {e}")

from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt') 
