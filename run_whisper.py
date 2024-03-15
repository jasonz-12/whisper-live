import sounddevice as sd
import numpy as np
import whisper
import queue

# Parameters
model_type = "base"  # Model type, can be tiny, base, small, medium, or large
sample_rate = 16000  # Whisper model's expected sample rate
chunk_size = 2048  # Size of chunks to read at a time (in samples)

# Load Whisper model
model = whisper.load_model(model_type)

# Initialize a queue to hold audio chunks
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This function is called for each audio chunk."""
    # Add the audio chunk to the queue
    audio_queue.put(indata.copy())

try:
    # Start streaming from microphone with the callback
    with sd.InputStream(callback=audio_callback, dtype='int16', channels=1, samplerate=sample_rate, blocksize=chunk_size):
        print("Transcribing... Press Ctrl+C to stop.")
        audio_buffer = np.array([], dtype='int16')
        
        while True:
            # Get an audio chunk from the queue
            chunk = audio_queue.get()
            audio_buffer = np.append(audio_buffer, chunk)
            
            if len(audio_buffer) >= sample_rate * 5:  # Process every ~5 seconds of audio
                # Convert to the format Whisper expects (float32, range [-1, 1])
                audio_float32 = audio_buffer.astype(np.float32) / 32768.0
                
                # Transcribe the audio chunk
                result = model.transcribe(audio_float32, temperature=0)
                print(result["text"])
                
                # Clear the buffer
                audio_buffer = np.array([], dtype='int16')

except KeyboardInterrupt:
    print("\nStopped.")
