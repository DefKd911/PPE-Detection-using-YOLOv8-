# PPE Detection 🚧🦺

This project implements a two-stage object detection pipeline using [YOLOv8](https://docs.ultralytics.com/) for detecting persons and their Personal Protective Equipment (PPE). Developed as part of a technical assessment at Syook.

---

## 🧠 Problem Statement

1. Convert Pascal VOC annotations to YOLOv8 format.
2. Train a **YOLOv8 model** for **person detection**.
3. Train another YOLOv8 model for **PPE detection** on cropped person images.
4. Write an inference pipeline that:
   - Detects persons on full images.
   - Crops and detects PPE on each detected person.
   - Converts PPE predictions back to full-image coordinates.
5. Draws bounding boxes using **OpenCV (cv2.rectangle + cv2.putText)**.

---

## 📁 Repository Structure

- `scripts/pascalVOC_to_yolo.py`: VOC → YOLO format converter.
- `scripts/inference.py`: End-to-end inference using both person + PPE models.
- `weights/`: Contains trained YOLOv8 model weights.
- `report/`: Final project report with methodology and metrics.
- `data/`: Sample images and labels (if needed).

---

## 🧪 Models Used

- YOLOv8m for both Person and PPE detection
- Trained on custom annotations derived from the given dataset.

---

![image](https://github.com/user-attachments/assets/ca1e4506-ca5b-46fc-b6ea-3115ba7d823e)
![001503](https://github.com/user-attachments/assets/2930fd3a-f934-496c-9398-0e121fa510cf)

## ⚙️ Setup & Inference

```bash
pip install -r requirements.txt

# Inference example:
python scripts/inference.py \
  --input_dir data/samples \
  --output_dir outputs \
  --person_det_model weights/person.pt \
  --ppe_detection_model weights/ppe.pt
