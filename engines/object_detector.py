import cv2
from ultralytics import YOLO 
class ObjectDetector: # model ko load karte hai jise memory mein rhaye for every image
  def __init__(self): 
    print("Loading YOLO model....")
    self.model = YOLO("yolov8n.pt") # load the YOLO model and n ka mtlb hai nano fastest and smallest version 
    print("Object detector ready!")

  def process(self, image_path):
    results = self.model(image_path) # ye image ko yolo model ko deta hai scan karta hai puri image return karta hai joh bhi ushe milta hai
    detections = [] 
    for result in results:
      for box in result.boxes: # haar result ke pass ekk box hai
        label = self.model.names[int(box.cls)] # box.cls is a number like 0,1,2 model.names is a dictionary joh convert karta hai inh numbers ko words mein like persons, door etc
        confidence = round(float(box.conf), 2)
        bbox = box.xyxy[0].tolist()
        
        if confidence > 0.3:
          detections.append({
            "label": label, 
            "confidence": confidence,
            "bbox": bbox
          })
    return{
      "detection": detections,
      "count": len(detections),
      "summary": ", ".join([d["label"] for d in detections])
    }