import cv2
import easyocr
import numpy as np
from utils.image_utils import resize_for_ocr

def preprocess_image(image_path):
  image = cv2.imread(image_path) # image ko load karto hai 
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray scale me convert kar diya
  gray = cv2.equalizeHist(gray) # contrast bda diya jishe dark hallway mein joh photos hongi use bhi detect ache se kar paye
  denoised = cv2.fastNlMeansDenoising(gray, h=30) # remove noise and grains
  resized = resize_for_ocr(denoised)
  return resized

class OCREngine:
  def __init__(self):
    print("Loading OCR model...")
    self.reader = easyocr.Reader(['en'])
    print("OCR model ready!")

  def process(self, image_path):
    preprocessed = preprocess_image(image_path)
    results = self.reader.readtext(preprocessed)

    detections = []
    for (bbox, text, confidence) in results:
      if confidence > 0.2:
        detections.append({
          "text": text,
          "confidence": round(confidence, 2),
          "bbox" : bbox
        })
    return {
      "raw_text": " ".join([d["text"] for d in detections]),
      "detections": detections,
      "count": len(detections)
    }