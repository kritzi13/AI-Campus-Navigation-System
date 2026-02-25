import cv2
import pytesseract
import numpy as np
import threading
import queue
import re
import time
import subprocess

# ── Mac Tesseract path (uncomment if needed) ───────────────
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# ── Native Mac TTS via `say` command ──────────────────────
speech_queue = queue.Queue()

def speak_mac(text):
    try:
        # -r 150 = speaking rate, -v Samantha = clear English voice
        subprocess.run(["say", "-r", "150", "-v", "Samantha", text],
                       check=True, timeout=30)
    except subprocess.TimeoutExpired:
        print("[TTS] Timeout — skipping")
    except Exception as e:
        print(f"[TTS Error] {e}")

def speech_worker():
    # Drain stale items so queue never backs up
    while True:
        text = speech_queue.get()
        if text is None:
            break
        # If newer text is already waiting, skip this one
        if speech_queue.empty():
            speak_mac(text)
        else:
            print(f"[TTS] Skipped (queue backed up): {text[:40]}")

threading.Thread(target=speech_worker, daemon=True).start()

# ── Startup test ───────────────────────────────────────────
print("[TTS] Testing Mac speech...")
speech_queue.put("Speech engine ready")

# ── Preprocessing ──────────────────────────────────────────
def preprocess(frame):
    resized = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed

# ── Text Validation ────────────────────────────────────────
def is_real_word(word):
    if len(word) < 2:
        return False
    if not re.search(r'[a-zA-Z]', word):
        return False
    non_alnum = sum(1 for c in word if not c.isalnum())
    if non_alnum / len(word) > 0.4:
        return False
    letters = re.sub(r'[^a-zA-Z]', '', word)
    if len(letters) >= 3:
        vowels = sum(1 for c in letters.lower() if c in 'aeiou')
        if vowels == 0:
            return False
    if len(set(word.lower())) <= 2 and len(word) > 3:
        return False
    return True

def clean_line(line):
    line = line.encode('ascii', 'ignore').decode()
    line = ' '.join(line.split())
    return line

def is_real_line(line):
    words = line.split()
    if not words:
        return False
    real_words = [w for w in words if is_real_word(w)]
    if len(real_words) / len(words) < 0.5:
        return False
    if len(real_words) == 1 and len(real_words[0]) < 4:
        return False
    return True

# ── OCR Config ─────────────────────────────────────────────
TESS_CONFIG = r'--oem 3 --psm 6 -l eng'

# ── Background OCR ─────────────────────────────────────────
ocr_lines = []
ocr_lock = threading.Lock()
ocr_busy = False

def run_ocr(frame):
    global ocr_busy
    processed = preprocess(frame)
    data = pytesseract.image_to_data(
        processed,
        config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    lines = {}
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        conf = int(data["conf"][i])
        if txt and conf > 55 and is_real_word(txt):
            line_id = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if line_id not in lines:
                lines[line_id] = txt
            else:
                lines[line_id] += " " + txt

    valid_lines = []
    for line in lines.values():
        cleaned = clean_line(line)
        if cleaned and is_real_line(cleaned):
            valid_lines.append(cleaned)

    with ocr_lock:
        ocr_lines.clear()
        ocr_lines.extend(valid_lines)

    ocr_busy = False

# ── Smart Speech Tracker ───────────────────────────────────
class SpeechTracker:
    def __init__(self, cooldown=8.0, stability_threshold=3):
        self.spoken_lines = {}
        self.line_seen_count = {}
        self.cooldown = cooldown
        self.stability_threshold = stability_threshold

    def should_speak(self, lines):
        now = time.time()
        to_speak = []

        for line in lines:
            self.line_seen_count[line] = self.line_seen_count.get(line, 0) + 1
            if self.line_seen_count[line] < self.stability_threshold:
                continue
            last_time = self.spoken_lines.get(line, 0)
            if now - last_time > self.cooldown:
                to_speak.append(line)
                self.spoken_lines[line] = now

        visible = set(lines)
        gone_lines = [l for l in self.line_seen_count if l not in visible]
        for l in gone_lines:
            del self.line_seen_count[l]
            if l in self.spoken_lines:
                del self.spoken_lines[l]

        return to_speak

tracker = SpeechTracker(cooldown=8.0, stability_threshold=3)

# ── Main Loop ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
print("Live OCR running — Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    frame_count += 1

    if frame_count % 25 == 0 and not ocr_busy:
        ocr_busy = True
        threading.Thread(target=run_ocr, args=(frame.copy(),), daemon=True).start()

    with ocr_lock:
        current_lines = list(ocr_lines)

    y = 40
    for line in current_lines:
        cv2.putText(frame, line[:80], (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 35

    if frame_count % 25 == 0 and current_lines:
        lines_to_speak = tracker.should_speak(current_lines)
        if lines_to_speak:
            speech_queue.put(". ".join(lines_to_speak))

    # Show what's queued for speech
    cv2.putText(frame, f"Frame {frame_count} | {'OCR running...' if ocr_busy else 'OCR idle'}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.imshow("Live OCR Reader", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()