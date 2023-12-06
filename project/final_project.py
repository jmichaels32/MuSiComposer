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

    output_dir = 'musicgen_generated_audio'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for audio_file in audio_files[:num_audio_files]:
        filenames.append(audio_file.replace('.mp3', ''))
        melody, sr = torchaudio.load(os.path.join(raw_audio_dir, audio_file))
        if melody.shape[0] > 1:
            melody = torch.mean(melody, dim=0, keepdim=True)
        wav = model.generate_with_chroma([description], melody.unsqueeze(0), sr)
        audio_file = filenames[-1]
        audio_write(os.path.join(output_dir, f'{audio_file}_melody'), wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    
# FIRST MODEL:
# Vision Transformer to Vision Transformer heads
# Inputs:
#   Spectrogram png data
#
# Outputs:
#   Raw Audio
import torch
import torch.nn as nn
import os
import random
from torchvision.models import vit_b_16
from audiocraft.models import WaveGAN
from PIL import Image
from torchvision.transforms import ToTensor

class MultiTaskedWaveGAN(nn.Module):
    def __init__(self, num_instruments=4):
        super(MultiTaskedWaveGAN, self).__init__()
        # Load the pre-trained Vision Transformer model
        self.vit = vit_b_16(pretrained=True)
        # Remove the classification head
        self.vit.heads = nn.Identity()
        # Make all the parameters trainable
        for param in self.vit.parameters():
            param.requires_grad = True
        
        # Initialize a WaveGAN model for each instrument
        self.wavegan_heads = nn.ModuleList([WaveGAN() for _ in range(num_instruments)])
        self.num_instruments = num_instruments
        self.to_tensor = ToTensor()

    def forward(self, x):
        # Forward pass through the transformer
        x = self.vit(x)
        # Generate audio for each instrument using the corresponding WaveGAN model
        outputs = [wavegan_head(x) for wavegan_head in self.wavegan_heads]
        return outputs

    def load_spectrogram(self, spectrogram_path):
        spectrogram_image = Image.open(spectrogram_path).convert('RGB')
        return self.to_tensor(spectrogram_image)

    def load_gold_labels(self, selected_chunk, instrument_audio_dir='instrument_audio'):
        gold_labels = []
        for instrument in ['bass', 'piano', 'other', 'drums']:
            instrument_file = f"{selected_chunk}_{instrument}.wav"
            instrument_path = os.path.join(instrument_audio_dir, instrument_file)
            if os.path.exists(instrument_path):
                waveform, _ = torchaudio.load(instrument_path)
                gold_labels.append(waveform)
            else:
                raise FileNotFoundError(f"Gold label for {instrument} not found in {selected_chunk}")
        return gold_labels

    def train_and_test(self, raw_audio_dir, spectrogram_dir, instrument_audio_dir, epochs=10, batch_size=16, learning_rate=0.001, test_split=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        # Get all audio file names, shuffle them, and split into train and test sets
        audio_files = [f.replace('.mp3', '') for f in os.listdir(raw_audio_dir) if f.endswith('.mp3')]
        random.shuffle(audio_files)
        split_idx = int(len(audio_files) * (1 - test_split))
        train_files = audio_files[:split_idx]
        test_files = audio_files[split_idx:]
        train_batches = [train_files[i:i + batch_size] for i in range(0, len(train_files), batch_size)]

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for audio_batch in train_batches:
                optimizer.zero_grad()
                batch_loss = 0
                for audio_file in audio_batch:
                    spectrogram_path = os.path.join(spectrogram_dir, f"{audio_file}.png")
                    spectrogram = self.load_spectrogram(spectrogram_path)
                    gold_labels = self.load_gold_labels(audio_file, instrument_audio_dir)
                    spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
                    outputs = self(spectrogram)
                    loss = sum([criterion(output, gold_label) for output, gold_label in zip(outputs, gold_labels)])
                    batch_loss += loss.item()
                batch_loss /= len(audio_batch)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_batches)}")

        # Save test file outputs
        self.eval()
        with torch.no_grad():
            for audio_file in test_files:
                spectrogram_path = os.path.join(spectrogram_dir, f"{audio_file}.png")
                spectrogram = self.load_spectrogram(spectrogram_path)
                spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
                outputs = self(spectrogram)
                for i, output in enumerate(outputs):
                    output_path = os.path.join('multitaskedwaveGAN_outputs', f"{audio_file}_instrument_{i}.wav")
                    torchaudio.save(output_path, output.cpu(), 16000)  # Assuming a sample rate of 16000 Hz

def train_multitaskedGAN():
    raw_audio_dir = './raw_audio'
    spectrogram_dir = './spectrograms'
    instrument_audio_dir = './instrument_audio'
    epochs = 10
    batch_size = 16
    learning_rate = 0.001
    test_split = 0.1

    model = MultiTaskedWaveGAN(num_instruments=4)
    model.train_and_test(raw_audio_dir, spectrogram_dir, instrument_audio_dir, epochs, batch_size, learning_rate, test_split)

# SECOND MODEL:
# Custon Audio Transformer to Diffusion heads
# Inputs:
#   Raw Audio .mp3 data
# 
# Outputs:
#   Raw Audio

# Call the training function with hyperparameters
#trained_model = train_model(batch_size=32, learning_rate=0.001, epochs=2)

# ------------
# Evaluation Functions
# ------------

def evaluate_song_quality(song_directory):
    """
    Evaluate the quality of songs in a given directory.

    Args:
        song_directory (str): The directory where songs are stored.

    Returns:
        dict: A dictionary with song filenames as keys and their quality score as values.
    """
    quality_scores = {}
    for song_file in os.listdir(song_directory):
        if song_file.endswith('.wav'):
            # Placeholder for actual quality evaluation logic
            quality_score = random.uniform(0, 10)  # Assign a random quality score between 0 and 10
            quality_scores[song_file] = quality_score
    return quality_scores

def evaluate_song_quality_PLACEHOLDER(song_directory):
    """
    Evaluate the quality of songs in a given directory.

    Args:
        song_directory (str): The directory where songs are stored.

    Returns:
        dict: A dictionary with song filenames as keys and their quality score as values.
    """
    quality_scores = {}
    for song_file in os.listdir(song_directory):
        if song_file.endswith('.wav'):
            # Placeholder for actual quality evaluation logic
            quality_score = random.uniform(0, 10)  # Assign a random quality score between 0 and 10
            quality_scores[song_file] = quality_score
    return quality_scores

# ------------
# MAIN SECTION
# ------------
# First Baseline
#music_gen(description='')

# Second Baseline
music_gen()

# First Model
#train_multitaskedGAN()