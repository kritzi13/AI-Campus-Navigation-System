import cv2
import easyocr
import numpy as np

def preprocess_image(image_path):
  image = cv2.imread(image_path) # image ko load karto hai 
  gray = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY) # gray scale me convert kar diya
  gray = cv2.equalizeHist(gray) # contrast bda diya jishe dark hallway mein joh photos hongi use bhi detect ache se kar paye
  denoised = cv2.fastNlMeansDenoising(gray, h=30) # remove noise and grains
  return denoised

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
      if confidence > 0.3:
        detections.sppend({
          "text": text,
          "confidence": round(confidence, 2),
          "bbox" : bbox
        })
    return {
      "raw_text": " ".join([d["text"] for d in detections]),
      "detections": detections,
      "count": len(detections)
    }