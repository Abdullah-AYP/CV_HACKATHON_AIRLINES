from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import sys
import shutil

# ================= 🚀 FINAL PUSH SETTINGS =================
MODEL_PATH = r"C:\Users\ZAH\Pictures\CV_HACKATHON_AIRLINES\Hack\Hackathon2_scripts\Hackathon2_scripts\runs\detect\runs\train\final_submission_model7\weights\best.pt"

# 1. THE LIMIT BREAKER
# Standard HD resolution. If this crashes, go back to 1600 immediately.
IMG_SIZE = 1920          

# 2. HYPER-SENSITIVE CONFIDENCE
# We want every single possible pixel accounted for.
CONF_THRESHOLD = 0.001   

# 3. PRECISION CLEANUP
# Lowering IOU to 0.5 helps merge duplicate boxes that hurt your score.
IOU_THRESHOLD = 0.5      

# 4. MAX SETTINGS
USE_TTA = True           
AGNOSTIC_NMS = True      
MAX_DET = 3000
# ==========================================================

if __name__ == '__main__': 
    print(f"🚀 STARTING 1920px RUN (IOU={IOU_THRESHOLD})...")
    
    if not os.path.exists(MODEL_PATH): sys.exit("❌ Model not found")
    model = YOLO(MODEL_PATH)
    
    this_dir = Path(__file__).parent
    yaml_path = this_dir / 'yolo_params.yaml'
    with open(yaml_path, 'r') as file: data = yaml.safe_load(file)
    images_dir = Path(data['test'])
    
    output_dir = this_dir / "predictions_submission_1920"
    if output_dir.exists(): shutil.rmtree(output_dir)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)

    # --- RUN PREDICTION ---
    image_files = [f for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not image_files: 
         image_files = [f for f in (images_dir / 'images').glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    print(f"📸 Processing {len(image_files)} images at {IMG_SIZE}px...")

    for i, img_path in enumerate(image_files):
        try:
            results = model.predict(
                source=img_path,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,     # <--- 1920px
                iou=IOU_THRESHOLD,  # <--- NEW TWEAK
                augment=USE_TTA,
                agnostic_nms=AGNOSTIC_NMS,
                max_det=MAX_DET,
                verbose=False
            )
            result = results[0]
            
            # Save Label
            out_txt = output_dir / 'labels' / img_path.with_suffix('.txt').name
            with open(out_txt, 'w') as f:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    x, y, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            print(f"   [{i+1}/{len(image_files)}] {img_path.name}", end='\r')

        except RuntimeError:
            print(f"\n❌ GPU OOM at 1920px! It was too much. Submitting the 1600px results is your best bet.")
            sys.exit()

    print(f"\n\n✅ DONE. Files in: {output_dir}")

    # --- FINAL VALIDATION ---
    print("\n📊 Calculating Final mAP (1920px)...")
    try:
        metrics = model.val(
            data=yaml_path, 
            split='test', 
            imgsz=IMG_SIZE, 
            conf=0.001, 
            iou=IOU_THRESHOLD,
            augment=USE_TTA,
            max_det=MAX_DET
        )
        print(f"\n🏆 Final mAP50:    {metrics.box.map50:.4f}")
        print(f"🏆 Final mAP50-95: {metrics.box.map:.4f}")
    except Exception as e:
        print(f"⚠️ Validation error: {e}")