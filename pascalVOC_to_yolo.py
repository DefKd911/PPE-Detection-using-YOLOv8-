import os
import xml.etree.ElementTree as ET
import argparse

def voc_to_yolo(input_dir, output_dir, classes):

    os.makedirs(output_dir, exist_ok=True)
    
    for xml_file in os.listdir(input_dir):
        if not xml_file.endswith(".xml"):
            continue
        
        tree = ET.parse(os.path.join(input_dir, xml_file))
        root = tree.getroot()
        
        
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        
        yolo_data = []
        
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()
            if class_name not in classes:
                continue  
            
            class_id = classes.index(class_name)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            
            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        
        
        txt_filename = os.path.join(output_dir, xml_file.replace(".xml", ".txt"))
        with open(txt_filename, "w") as f:
            f.write("\n".join(yolo_data))
    
    print(f"Conversion completed! YOLO annotations saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML annotations to YOLOv8 format.")
    parser.add_argument("input_dir", type=str, help="Path to directory containing Pascal VOC XML annotations.")
    parser.add_argument("output_dir", type=str, help="Path to directory where YOLO annotations will be saved.")
    
    args = parser.parse_args()
    
    CLASSES = ["person", "hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]
    
    voc_to_yolo(args.input_dir, args.output_dir, CLASSES)

if __name__ == "__main__":
    main()
