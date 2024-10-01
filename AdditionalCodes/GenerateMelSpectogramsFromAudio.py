import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch

import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch

# Load the audio file
audio_path = "sample_data/extracted_audio.wav"
y, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16 kHz

# Convert to Mel-spectrogram
# The arguments y and sr should be passed as keyword arguments
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Convert to dB scale (log scale)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Convert Mel-spectrogram to tensor (add batch dimension if necessary)
mel_spec_tensor = torch.tensor(mel_spec_db).unsqueeze(0)

# Show the Mel-spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-spectrogram')
plt.tight_layout()
plt.show()
