
Separate YOLOv8 Training Scripts — D1-D5
========================================

These scripts train one dataset at a time.

Dataset status assumed:
- D1 OK: classification, Container vs Wagon
- D2 OK: BAW-Sack detection with negative context images
- D3 OK: Plastiksack detection with negative context images
- D4 OK: Papier detection with negative context images
- D5 OK: 3-class detection with negative context images

Common baseline:
- image size: 1536
- no augmentation
- seed: 42
- patience: 50
- epochs: 200

D1:
- task: classification
- model: yolov8m-cls.pt
- batch: 4

D2-D5:
- task: detection
- model: yolov8m.pt
- batch: 1

Run one by one:

cd C:\Hadi\Foreign_Material\Code\Python\Jul26

python train_D1_Context_Classification_yolov8_noaug_img1536.py
python train_D2_BAW_Sack_yolov8_noaug_img1536.py
python train_D3_Plastiksack_yolov8_noaug_img1536.py
python train_D4_Papier_yolov8_noaug_img1536.py
python train_D5_Dominant_3class_yolov8_noaug_img1536.py

If you are using your venv explicitly:

C:\Hadi\venvs\fm-yolo11\Scripts\python.exe train_D1_Context_Classification_yolov8_noaug_img1536.py

If CUDA out of memory happens:
- For D1 reduce BATCH from 4 to 2 or 1.
- For D2-D5 BATCH is already 1; if it still fails, use IMG_SIZE = 1280 for a diagnostic run, but the preferred protocol is 1536.

First clean baseline:
- Keep USE_PREVIOUS_BEST_FROM_THIS_DATASET = False

Later improvement runs:
- Set USE_PREVIOUS_BEST_FROM_THIS_DATASET = True in the relevant script.
