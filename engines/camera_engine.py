import cv2
import time # we use it too control kitne baar process kiya hai frame
from engines.ocr_engine import OCREngine
from engines.object_detector import ObjectDetector

class CameraEngine:
  def __init__(self):
    print("Loading camera engine...")
    self.ocr = OCREngine() # Load OCR engine once into memory
    self.detector = ObjectDetector() # load yolo once into memory
    self.is_running = False # a flag to control when to stop the camera loop 
    self.frame_count = 0 # count karega frame joh hum baad mein use kar shake sirf 10 frame process karne ke liye
    print("Camera engine ready!") 

# frame_count ko hum use kare like count kare frame 1,2,3 or jab 10 pe phouchega toh process kare 10th wla and then reset to 0 frame jisse speed aur efficency bani rhaye
