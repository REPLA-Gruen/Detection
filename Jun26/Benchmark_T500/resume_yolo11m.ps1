& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"

yolo detect train `
    model=C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\yolo11m_T500_960\weights\last.pt `
    resume=True
