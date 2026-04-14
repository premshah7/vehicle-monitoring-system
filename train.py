from ultralytics import YOLO
import os

def train_model():
    model = YOLO('yolov8n.pt') 
    current_working_dir = os.getcwd()

    model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=640, 
        device=0, 
        batch=4,        
        workers=2,      
        amp=False,      
        project=current_working_dir, 
        name='runs/detect/train'
    )

if __name__ == "__main__":
    train_model()