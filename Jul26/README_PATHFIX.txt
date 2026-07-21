PATH-FIXED training scripts
===========================

Problem fixed:
The previous scripts created the run folder before model.train().
Ultralytics then detected that the folder already existed and created a folder with a suffix like "2".
The script then looked for best.pt in the original planned folder, not the actual folder.

Fix:
- Do not create the run folder before model.train().
- Save the return object from model.train().
- Read the actual save_dir from results.save_dir or model.trainer.save_dir.
- Write training_info.txt only after training into the actual run folder.

Use:
Copy these scripts into:
C:\Hadi\Foreign_Material\Code\Python\Jul26

For D1, no retraining is needed. Your existing D1 best.pt is already OK.
Use these fixed scripts for D2-D5 from now on.

Run:
python train_D2_BAW_Sack_yolov8_noaug_img1536_PATHFIX.py
python train_D3_Plastiksack_yolov8_noaug_img1536_PATHFIX.py
python train_D4_Papier_yolov8_noaug_img1536_PATHFIX.py
python train_D5_Dominant_3class_yolov8_noaug_img1536_PATHFIX.py
