import cv2
import argparse
import torch
import os
from ultralytics import YOLO

def convert_ppe_to_full_image(ppe_bbox, person_bbox, crop_w, crop_h):
    x1_ppe, y1_ppe, x2_ppe, y2_ppe = ppe_bbox  
    x1_person, y1_person, x2_person, y2_person = person_bbox  
    
    x1_full = x1_person + (x1_ppe / crop_w) * (x2_person - x1_person)
    y1_full = y1_person + (y1_ppe / crop_h) * (y2_person - y1_person)
    x2_full = x1_person + (x2_ppe / crop_w) * (x2_person - x1_person)
    y2_full = y1_person + (y2_ppe / crop_h) * (y2_person - y1_person)

    return int(x1_full), int(y1_full), int(x2_full), int(y2_full)

def run_inference(input_dir, output_dir, person_model_path, ppe_model_path):
    
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    
    ppe_class_names = ppe_model.names

    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        
        results_person = person_model(image)

        for result in results_person:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_person = image[y1:y2, x1:x2]

                
                results_ppe = ppe_model(cropped_person)
                crop_h, crop_w, _ = cropped_person.shape

                for ppe_result in results_ppe:
                    for ppe_box, cls_id, conf in zip(ppe_result.boxes.xyxy, 
                                                     ppe_result.boxes.cls, 
                                                     ppe_result.boxes.conf):
                        x1_ppe, y1_ppe, x2_ppe, y2_ppe = map(int, ppe_box.tolist())
                        class_id = int(cls_id.item())
                        confidence = float(conf.item())
                        class_name = ppe_class_names[class_id]

                        x1_full, y1_full, x2_full, y2_full = convert_ppe_to_full_image(
                            (x1_ppe, y1_ppe, x2_ppe, y2_ppe),
                            (x1, y1, x2, y2),
                            crop_w, crop_h
                        )

                        
                        label = f"{class_name} {confidence:.2f}"
                        cv2.rectangle(image, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)
                        cv2.putText(image, label, (x1_full, y1_full - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPE Detection Pipeline")
    parser.add_argument("input_dir", type=str, help="Directory of input images")
    parser.add_argument("output_dir", type=str, help="Directory to save results")
    parser.add_argument("person_det_model", type=str, help="Path to YOLO person detection model")
    parser.add_argument("ppe_detection_model", type=str, help="Path to YOLO PPE detection model")

    args = parser.parse_args()
    run_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)
