from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import tempfile
import os
import pyaudio
import wave
import numpy as np
from threading import Thread
from openai import OpenAI
import scipy.signal as signal
import webrtcvad
# VAD Constants
VAD_SAMPLE_RATE = 16000
VAD_FRAME_DURATION = 160  # milliseconds
VAD_PADDING_DURATION = 300  # milliseconds
VAD_MODE = 3

vad = webrtcvad.Vad(VAD_MODE)

# Constants for audio processing
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

client = OpenAI(api_key='key')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variable to hold audio data
audio_data = []

@app.route("/http-call")
def http_call():
    """return JSON with string data as the value"""
    data = {'data': 'This text was fetched using an HTTP call to server on render'}
    return jsonify(data)

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print("client has connected")

def detect_speech(audio_np):
    """Detects speech in audio using WebRTC VAD"""
    speech = False
    frames = []
    val=webrtcvad.valid_rate_and_frame_length(16000, 160)
    print(val)
    # Convert audio to 16kHz
    audio_16k = signal.resample(audio_np, len(audio_np) * VAD_SAMPLE_RATE // RATE)
    print(audio_16k)
    # Pad the audio to ensure all frames are of equal length
    padding = bytes(VAD_PADDING_DURATION * VAD_SAMPLE_RATE // 1000)
    audio_padded = audio_16k.tobytes() + padding

    # Split audio into frames
    for i in range(0, len(audio_padded), VAD_FRAME_DURATION * VAD_SAMPLE_RATE // 1000):
        frames.append(audio_padded[i:i + VAD_FRAME_DURATION * VAD_SAMPLE_RATE // 1000])
        vad.set_mode(1)
    print("Frames: ", frames)
    print("VAD_SAMPLE_RATE: ", VAD_SAMPLE_RATE)
    # Process frames
    for frame in frames:
        if vad.is_speech(frame, VAD_SAMPLE_RATE):
            speech = True
            break
    
    
    return speech

@socketio.on('data')
def handle_message(data):
    global audio_data
    try:
        # Decode base64 audio data
        decoded_audio_data = base64.b64decode(data['file'])
        
        # Convert the audio data into a numpy array
        audio_np = np.frombuffer(decoded_audio_data, dtype=np.int16)
        
        # Append the new audio data to the global audio data list
        audio_data.extend(audio_np)

        # Process audio in chunks of CHUNK samples (1024 samples)
        if len(audio_data) >= RATE * 5:  # Process every 5 seconds of audio
            # Extract 5 seconds of audio data
            audio_np = np.array(audio_data[:RATE * 5])
            audio_data = audio_data[CHUNK * 5:]

            # Check if the audio contains speech
            if detect_speech(audio_np):
                # Write binary data into a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio_file:
                    wf = wave.open(tmp_audio_file, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(audio_np))
                    wf.close()
                    audio_path = tmp_audio_file.name

                # Send the WAV file to the Whisper AI for transcription
                transcription = client.audio.transcriptions.create(
                    file=open(audio_path, 'rb'),
                    model=data['model'],
                    language=data.get('language', 'en')
                )
                
                print("Transcripted text: ", transcription.text)
                # Emit the transcription result back to the client
                emit("transcription", {
                    "status": True,
                    "message": "Transcripted text: ",
                    "text": transcription.text
                })
                # Clean up temporary file
                # os.remove(audio_path)
            else:
                print("No speech detected in the audio")
    
    except Exception as e:
        print(e)
        # Handle the error or emit it back to the client

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects from the server"""
    print("user disconnected")
    emit("disconnect", f"user {request.sid} disconnected", broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)
