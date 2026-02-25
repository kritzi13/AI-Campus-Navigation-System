import tkinter as tk
import pyttsx3
import speech_recognition as sr
import subprocess
import threading
import sys

# ---------- Text to Speech ----------
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------- Open Files ----------
def open_notice():
    subprocess.Popen([sys.executable, "text.py"])

def open_navigation():
    subprocess.Popen([sys.executable, "try.py"])

# ---------- Voice Command ----------
def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Please say notice or navigation")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print("User said:", command)

            if "notice" in command:
                speak("Opening notices")
                open_notice()

            elif "navigation" in command:
                speak("Opening navigation")
                open_navigation()

            else:
                speak("Sorry, I did not understand")

        except:
            speak("Could not understand your voice")

# ---------- GUI ----------
root = tk.Tk()
root.title("PathMate")
root.geometry("600x400")
root.configure(bg="black")  # high contrast

# ---------- Heading ----------
heading = tk.Label(
    root,
    text="Welcome to your PathMate",
    font=("Arial", 22, "bold"),
    fg="white",
    bg="black"
)
heading.pack(pady=40)

# ---------- Instruction ----------
instruction = tk.Label(
    root,
    text="Say or choose: Notice or Navigation",
    font=("Arial", 14),
    fg="white",
    bg="black"
)
instruction.pack(pady=10)

# ---------- Buttons ----------
btn_notice = tk.Button(
    root,
    text="Notice",
    font=("Arial", 16),
    width=15,
    height=2,
    command=open_notice
)
btn_notice.pack(pady=10)

btn_navigation = tk.Button(
    root,
    text="Navigation",
    font=("Arial", 16),
    width=15,
    height=2,
    command=open_navigation
)
btn_navigation.pack(pady=10)

# ---------- Speak on Open ----------
def welcome():
    speak("Welcome to your PathMate. Do you want to read notices or want navigation?")
    threading.Thread(target=listen_command).start()

root.after(1000, welcome)

root.mainloop()