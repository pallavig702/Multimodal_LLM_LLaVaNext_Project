#pip install git+https://github.com/openai/whisper.git
import whisper

# Load pre-trained Whisper model
model = whisper.load_model("base")

# Transcribe audio file to text
audio_path = "sample_data/extracted_audio.wav"
result = model.transcribe(audio_path)

# Extract transcribed text
transcribed_text = result['text']
print("Transcribed Text:", transcribed_text)
