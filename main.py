from engines.ocr_engine import OCREngine
from engines.object_detector import ObjectDetector

# Load both engines once
ocr = OCREngine()
detector = ObjectDetector()

image_path = "test_images/sign_board.jpeg"

# Run both
ocr_result = ocr.process(image_path)
yolo_result = detector.process(image_path)

print("=== OCR RESULTS ===")
print("Text found:", ocr_result["raw_text"])
print("Total detections:", ocr_result["count"])

print("\n=== YOLO RESULTS ===")
print("Objects found:", yolo_result["summary"])
print("Total objects:", yolo_result["count"])
for obj in yolo_result["detections"]:
    print(f"  → {obj['label']} (confidence: {obj['confidence']})")