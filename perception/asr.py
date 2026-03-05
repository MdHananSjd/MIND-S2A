import os
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile

# Load Whisper model once (important for performance)
MODEL = whisper.load_model("base")


def transcribe_from_file(path: str) -> str:
    """
    Takes audio file path and returns transcribed text using Whisper base model.
    """
    if not os.path.exists(path):
        error_msg = f"Error: Audio file not found at {path}"
        print(error_msg)
        return error_msg

    try:
        result = MODEL.transcribe(path)
        transcribed_text = result.get("text", "").strip()

        print("\n[Layer 1: ASR Output]")
        print(f"Transcribed Text: {transcribed_text}\n")

        return transcribed_text

    except Exception as e:
        error_msg = f"Error during file transcription: {str(e)}"
        print(error_msg)
        return error_msg


def transcribe_from_mic(duration: int = 5) -> str:
    """
    Records audio from the microphone for a set duration and transcribes it.
    """
    fs = 16000  # Sample rate

    print(f"Recording for {duration} seconds...")
    
    try:
        # Record as float32 for better compatibility
        recording = sd.rec(int(duration * fs),
                           samplerate=fs,
                           channels=1,
                           dtype='float32')
        sd.wait()
        print("Recording finished.")

        # Normalize (safety step)
        recording = np.squeeze(recording)

        # Save to temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        wav.write(temp_path, fs, recording)

        try:
            text = transcribe_from_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return text

    except Exception as e:
        error_msg = f"Error during mic recording/transcription: {str(e)}"
        print(error_msg)
        return error_msg


def transcribe() -> str:
    """
    Standard entry point for ASR. Defaults to transcribing from the microphone.
    """
    return transcribe_from_mic()


if __name__ == "__main__":
    # Test mic recording
    transcribe_from_mic(5)