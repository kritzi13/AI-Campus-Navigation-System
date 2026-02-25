import cv2
import pytesseract
import pyttsx3
from PIL import Image
import time

# -------- Initialize TTS ONCE --------
engine = pyttsx3.init()
engine.setProperty('rate', 155)

def speak(text):
    engine.say(text)

# -------- Image Preprocessing --------
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        speak("Image not found. Please check the image path.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh

# -------- OCR + Speech --------
def image_to_speech(image_path):
    processed_img = preprocess_image(image_path)

    if processed_img is None:
        return

    pil_img = Image.fromarray(processed_img)

    text = pytesseract.image_to_string(pil_img)

    print("\nExtracted Text:\n")
    print(text)

    if text.strip() == "":
        speak("Sorry, I could not find any readable text.")
        return

    speak("Reading the text from the image")

    for line in text.split("\n"):
        line = line.strip()
        if line:
            speak(line)

# -------- MAIN --------
if __name__ == "__main__":
    image_path = "/Users/karansingh22/Documents/minproject/t3qWG.png"

    image_to_speech(image_path)

    # 🔑 IMPORTANT: Keep engine alive until all speech is done
    engine.runAndWait()
    time.sleep(0.5)