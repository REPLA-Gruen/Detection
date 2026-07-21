
D2 AI module: YOLO candidate + BAW-Sack crop verifier
=====================================================

This replaces the weak single-stage D2 detector with a two-stage module:

1. YOLO D2 detector works as a candidate generator.
   Use low confidence to preserve recall.

2. A crop verifier classifier checks each candidate crop and learns
   BAW-Sack visual evidence such as the X/cross pattern and sack texture.

Run order
---------

Copy all scripts to:
C:\Hadi\Foreign_Material\Code\Python\Jul26\D2_BAW_Sack_Detection

Then:

cd C:\Hadi\Foreign_Material\Code\Python\Jul26

python .\D2_BAW_Sack_Detection\create_D2_BAW_Sack_crop_verifier_dataset.py

python .\D2_BAW_Sack_Detection\train_D2_BAW_Sack_crop_verifier_yolov8cls.py

After classifier training, copy the printed classifier best.pt path.

Open:
D2_AI_module_yolo_candidate_plus_crop_verifier.py

Paste:
DETECTOR_MODEL = Path(r"...D2 detector...\weights\best.pt")
VERIFIER_MODEL = Path(r"...crop verifier...\weights\best.pt")

Recommended first detector path:
C:\Hadi\Foreign_Material\Runs\Focused_Datasets_YOLOv8_NoAug_Img1536_Separate\D2_BAW_Sack_binary_neg1to1\D2_yolov8m_detect_noaug_img1536_neg1to1_20260710_015710\weights\best.pt

Then run:

python .\D2_BAW_Sack_Detection\D2_AI_module_yolo_candidate_plus_crop_verifier.py

Outputs:
- candidate CSV with detector confidence and verifier score
- threshold sweep summary
- rendered final detections
