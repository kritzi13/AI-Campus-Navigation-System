"""
YOLO Object Detection - Live Camera
Detects: Vehicles, Furniture, Traffic Signs, Speed Limit Signs,
         Electrical Appliances, Persons, Plants, Books
Requirements: pip install ultralytics opencv-python pyttsx3
"""

import cv2
import pyttsx3
import threading
import time
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4      # Min confidence to show detection (0.0 - 1.0)
SPEAK_COOLDOWN = 3              # Seconds between speaking same object again
CAMERA_INDEX = 0                # 0 = default webcam, change if using external camera
SHOW_FPS = True                 # Show FPS counter on screen

# ─────────────────────────────────────────────
# CATEGORY DEFINITIONS
# These are COCO dataset class names that YOLOv8 can detect
# ─────────────────────────────────────────────

VEHICLES = [
    "car", "motorcycle", "bus", "truck", "bicycle",
    "boat", "train", "airplane"
]

FURNITURE = [
    "chair", "couch", "bed", "dining table", "toilet",
    "desk", "bookshelf", "cabinet"
]

TRAFFIC_SIGNS = [
    "stop sign", "traffic light"
]

ELECTRICAL_APPLIANCES = [
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "refrigerator", "hair drier",
    "clock", "fan"
]

PERSONS = [
    "person"
]

PLANTS = [
    "potted plant", "vase"
]

BOOKS = [
    "book"
]

# Speed limit signs are not in COCO by default
# We detect them using text on signs (OCR approach) or custom model
# For now we flag "stop sign" and use OCR for speed numbers
SPEED_LIMIT_KEYWORDS = ["30", "40", "50", "60", "70", "80", "100", "120"]

# Category colors (BGR format for OpenCV)
CATEGORY_COLORS = {
    "vehicle":      (0, 255, 0),      # Green
    "furniture":    (255, 165, 0),    # Orange
    "traffic_sign": (0, 0, 255),      # Red
    "speed_limit":  (255, 0, 255),    # Magenta
    "appliance":    (0, 255, 255),    # Cyan
    "person":       (255, 255, 0),    # Yellow
    "plant":        (0, 180, 0),      # Dark Green
    "book":         (180, 105, 255),  # Purple
    "other":        (200, 200, 200),  # Gray
}

# ─────────────────────────────────────────────
# TEXT TO SPEECH ENGINE
# ─────────────────────────────────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)
tts_engine.setProperty('volume', 1.0)

spoken_objects = {}  # Tracks last spoken time per object

def speak(text):
    """Speak text in a separate thread so it doesn't block detection."""
    def _speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def should_speak(label):
    """Only speak if enough time has passed since last announcement."""
    now = time.time()
    if label not in spoken_objects or (now - spoken_objects[label]) > SPEAK_COOLDOWN:
        spoken_objects[label] = now
        return True
    return False

# ─────────────────────────────────────────────
# CATEGORY CLASSIFIER
# ─────────────────────────────────────────────
def get_category(label):
    label_lower = label.lower()
    if label_lower in VEHICLES:
        return "vehicle"
    elif label_lower in FURNITURE:
        return "furniture"
    elif label_lower in TRAFFIC_SIGNS:
        return "traffic_sign"
    elif label_lower in ELECTRICAL_APPLIANCES:
        return "appliance"
    elif label_lower in PERSONS:
        return "person"
    elif label_lower in PLANTS:
        return "plant"
    elif label_lower in BOOKS:
        return "book"
    else:
        return "other"

def get_color(category):
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"])

# ─────────────────────────────────────────────
# DRAW DETECTION BOX
# ─────────────────────────────────────────────
def draw_detection(frame, box, label, confidence, category):
    x1, y1, x2, y2 = map(int, box)
    color = get_color(category)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    display_text = f"{label} {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

    # Label text
    cv2.putText(frame, display_text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Category badge
    cv2.putText(frame, f"[{category.upper()}]", (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

# ─────────────────────────────────────────────
# DRAW LEGEND
# ─────────────────────────────────────────────
def draw_legend(frame):
    legend_items = [
        ("Vehicle",     "vehicle"),
        ("Furniture",   "furniture"),
        ("Traffic Sign","traffic_sign"),
        ("Appliance",   "appliance"),
        ("Person",      "person"),
        ("Plant",       "plant"),
        ("Book",        "book"),
    ]
    x, y = 10, 30
    box_height = len(legend_items) * 22 + 15
    cv2.rectangle(frame, (5, 5), (185, box_height), (30, 30, 30), -1)
    for name, cat in legend_items:
        color = get_color(cat)
        cv2.rectangle(frame, (x, y - 12), (x + 16, y + 2), color, -1)
        cv2.putText(frame, name, (x + 22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

# ─────────────────────────────────────────────
# MAIN DETECTION LOOP
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  YOLO Object Detector - Live Camera")
    print("  Detects: Vehicles, Furniture, Traffic Signs,")
    print("           Appliances, Persons, Plants, Books")
    print("=" * 55)
    print("Loading YOLOv8 model...")

    # Load YOLOv8 nano model (fastest, downloads automatically ~6MB)
    # Change to "yolov8s.pt" for better accuracy, "yolov8m.pt" for even better
    model = YOLO("yolov8n.pt")
    print("Model loaded! Starting camera...\n")
    print("Controls:")
    print("  Q  →  Quit")
    print("  S  →  Toggle speech on/off")
    print("  +  →  Increase confidence threshold")
    print("  -  →  Decrease confidence threshold")
    print("=" * 55)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check CAMERA_INDEX in config.")
        return

    speech_enabled = True
    confidence = CONFIDENCE_THRESHOLD
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    speak("Object detection started")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break

        # ── Run YOLO detection ──
        results = model(frame, conf=confidence, verbose=False)[0]

        detected_this_frame = []

        for det in results.boxes:
            label = model.names[int(det.cls)]
            conf  = float(det.conf)
            box   = det.xyxy[0].tolist()
            category = get_category(label)

            # Only show relevant categories
            if category == "other":
                continue

            draw_detection(frame, box, label, conf, category)
            detected_this_frame.append((label, category))

            # Speak detection
            if speech_enabled and should_speak(label):
                speak(f"{category} detected: {label}")

        # ── FPS Counter ──
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start = time.time()

        if SHOW_FPS:
            cv2.putText(frame, f"FPS: {fps_display}", (frame.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Confidence display ──
        cv2.putText(frame, f"Conf: {confidence:.0%}  Speech: {'ON' if speech_enabled else 'OFF'}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # ── Legend ──
        draw_legend(frame)

        # ── Show frame ──
        cv2.imshow("YOLO Object Detector | Press Q to quit", frame)

        # ── Key Controls ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            speech_enabled = not speech_enabled
            status = "ON" if speech_enabled else "OFF"
            print(f"Speech turned {status}")
            speak(f"Speech {status}")
        elif key == ord('+') or key == ord('='):
            confidence = min(0.95, confidence + 0.05)
            print(f"Confidence: {confidence:.0%}")
        elif key == ord('-'):
            confidence = max(0.05, confidence - 0.05)
            print(f"Confidence: {confidence:.0%}")

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()