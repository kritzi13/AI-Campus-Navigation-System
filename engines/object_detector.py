import cv2
from ultralytics import YOLO 
class ObjectDetector: # model ko load karte hai jise memory mein rhaye for every image
  def __init__(self): 
    print("Loading YOLO model....")
    self.model = YOLO("yolov8n.pt") # load the YOLO model and n ka mtlb hai nano fastest and smallest version 
    print("Object detector ready!")

  def process(self, image_path):
        results = self.model(image_path)
        
        detections = []
        for result in results:
            for box in result.boxes:
                label = self.model.names[int(box.cls)]
                confidence = round(float(box.conf), 2)
                bbox = box.xyxy[0].tolist()
                
                if confidence > 0.3:
                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "bbox": bbox
                    })
        
        return {
            "detections": detections,
            "count": len(detections),
            "summary": ", ".join([d["label"] for d in detections])
        }