import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import torch

# 2 BASELINE MODELS:
# Text prompt for MusicGEN "a lofi study song" using musicgen-melody (melody and text descriptions)
# No text prompt for MusicGEN from Facebook/Meta
def music_gen(description='a lofi study song', num_audio_files=100):
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.set_generation_params(duration=8)  # generate 8 seconds.

    raw_audio_dir = './raw_audio'
    audio_files = [f for f in os.listdir(raw_audio_dir) if f.endswith('.mp3')]
    melodies = []
    sample_rates = []
    filenames = []

    for audio_file in audio_files[:num_audio_files]:
        filenames.append(audio_file.replace('.mp3', ''))
        melody, sr = torchaudio.load(os.path.join(raw_audio_dir, audio_file))
        if melody.shape[0] > 1:
            melody = torch.mean(melody, dim=0, keepdim=True)
        melodies.append(melody)
        sample_rates.append(sr)

    wavs = []
    for melody, sr in zip(melodies, sample_rates):
        wav = model.generate_with_chroma([description], melody.unsqueeze(0), sr)
        wavs.append(wav)
    wavs = torch.cat(wavs, dim=0)

    output_dir = 'musicgen_generated_audio'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, one_wav in enumerate(wavs):
        # Will save under {idx}_melody.wav, with loudness normalization at -14 db LUFS.
        audio_file = filenames[idx]
        audio_write(os.path.join(output_dir, f'{audio_file}_melody'), one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    
# FIRST MODEL:
# vggish embedding layer with multi-layer transformer heads (using spleeter for separation)




# FIRST MODEL:
# Vision Transformer to Vision Transformer heads
# Inputs:
#   Spectrogram png data
#
# Outputs:
#   Raw Audio



# SECOND MODEL:
# Custon Audio Transformer to Diffusion heads
# Inputs:
#   Raw Audio .mp3 data
# 
# Outputs:
#   Raw Audio

import torch
import os
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, audio_dir='./raw_audio'):
        self.audio_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith('.mp3')]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform

class NaiveRegression(torch.nn.Module):
    def __init__(self):
        super(NaiveRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=16000, out_features=1)  # Assuming 1 second of audio at 16kHz

    def forward(self, x):
        return self.linear(x)

def train_model(batch_size=32, learning_rate=0.001, epochs=2):
    audio_dataset = AudioDataset()
    data_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)

    model = NaiveRegression()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Example of training loop
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, torch.zeros(outputs.size()))  # Naive example with zero labels
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}')

    return model

# Call the training function with hyperparameters
#trained_model = train_model(batch_size=32, learning_rate=0.001, epochs=2)

# ------------
# MAIN SECTION
# ------------
# First Baseline
#music_gen(description='')

# Second Baseline
#music_gen()

# First Model
