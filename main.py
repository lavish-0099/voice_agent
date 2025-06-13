from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pyttsx3
import os
import uuid
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import json
from vosk import Model, KaldiRecognizer
import requests
import time
from dotenv import load_dotenv

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")


vosk_model_path = "vosk-model"
if not os.path.exists(vosk_model_path):
    raise Exception(f"Vosk model not found at '{vosk_model_path}'. Download from https://alphacephei.com/vosk/models")

model = Model(vosk_model_path)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/talk")
async def talk():
    filename = f"temp_{uuid.uuid4().hex}.wav"

 #Record audio
    duration = 5
    fs = 16000
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, recording)
    print("Recording done.")
    
# Transcribe with Vosk
    
    rec = KaldiRecognizer(model, fs)
    with open(filename, "rb") as f:
        audio_data = f.read()
    os.remove(filename)

    if rec.AcceptWaveform(audio_data):
        result = json.loads(rec.Result())
        text = result.get("text", "").strip()
    else:
        text = ""

    if not text:
        return {"error": "Could not understand your speech."}

    #Call Togeher AI
    try:
        start_time = time.time()
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.7
        }

        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"].strip()
        else:
            reply = f"Together API Error: {response.text}"

        latency = round(time.time() - start_time, 2)

    except Exception as e:
        reply = f"Error from Together AI: {str(e)}"
        latency = 0

    # Speak the reply
    engine = pyttsx3.init()
    engine.say(reply)
    engine.runAndWait()

    return {
        "input": text,
        "reply": reply,
        "latency_seconds": latency
    }


#run command->  C:\Users\lavis\AppData\Roaming\Python\Python313\Scripts\uvicorn.exe main:app --reload
