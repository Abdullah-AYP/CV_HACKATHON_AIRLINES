from ultralytics import YOLO
from pathlib import Path
import os
import torch
import yaml

# ================= CONFIGURATION =================
EPOCHS = 50            # The "Winning" number for high accuracy
IMG_SIZE = 1024         # High resolution for small objects
BATCH_SIZE = 2          # Use 8 for Colab (T4), use 2 if on your RTX 5050
MODEL_NAME = "yolov8l.pt" # Large model
# =================================================

def main():
    # 1. Setup Paths
    this_dir = Path(__file__).parent
    os.chdir(this_dir)
    
    # Check for data.yaml
    yaml_path = this_dir / 'yolo_params.yaml' # Or 'data.yaml', check your file name!
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found. Please ensure your config file is here.")
        return

    # 2. Initialize Model
    print(f"--- STARTING FINAL TRAINING (1 Fold, {EPOCHS} Epochs) ---")
    model = YOLO(MODEL_NAME)

    # 3. Train (Standard Single Fold)
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        patience=30,       # Stop if no improvement for 30 epochs
        
        # Hardware Settings
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,          # Use first GPU
        workers=2,         # Low workers for stability
        
        # Challenge-Specific Augmentations (Lighting & Occlusion)
        optimizer='AdamW',
        lr0=0.0005,
        hsv_h=0.02,        # Handle varying lighting
        hsv_s=0.7, 
        hsv_v=0.4,
        mosaic=1.0,        # Critical for occlusions
        mixup=0.1,
        
        # Saving
        project='runs/train',
        name='final_submission_model'
    )
    
    print("\nTraining Complete.")
    print(f"Best model saved at: {this_dir / 'runs/train/final_submission_model/weights/best.pt'}")

if __name__ == '__main__':
    main()