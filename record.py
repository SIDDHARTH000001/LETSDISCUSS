import pyaudio
import wave
from pydub import AudioSegment
from datetime import datetime
import tempfile
import numpy as np
import nltk

def record_audio(file_name, duration=8, sample_rate=44100, chunk_size=1024):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    frames = []
    print("Recording...")
    for i in range(int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))


# if __name__ == "__main__":
#     input_file = "input_audio.mp3"
#     record_audio(f'Audio/Audio_{datetime.now().strftime("%H%M%S")}.mp3')
#     print(f"Audio saved...")
