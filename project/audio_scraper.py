import os
import subprocess
from pytube import YouTube
import io
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
from scipy.io import wavfile
import soundfile as sf
from PIL import Image

def download_and_split_audio(download_path='./raw_audio', chunk_duration=30):
    # Ensure download directory exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Load video links from file
    with open('music_youtube_urls.txt', 'r') as file:
        video_links = file.readlines()

    # Download and split audio files into chunks
    for i, link in enumerate(video_links[100:]):
        i += 100
        yt = YouTube(link.strip())
        audio_stream = yt.streams.get_audio_only()
        # Sanitize filename to avoid issues with special characters
        filename = f"{str(i)}.mp3"
        temp_filename = f"{str(i)}_temp.mp3"
        audio_stream.download(output_path=download_path, filename=temp_filename)
        
        # Load the temporary audio file
        temp_filepath = os.path.join(download_path, temp_filename)
        file_size = os.path.getsize(temp_filepath)
        if file_size >= 4 * 1024 * 1024 * 1024:
            os.remove(temp_filepath)
            continue
        audio = AudioSegment.from_file(temp_filepath)
        
        # Split the audio segment into 30-second chunks and pad the last one if necessary
        for j in range(0, len(audio), chunk_duration * 1000):
            chunk = audio[j:j + chunk_duration * 1000]
            if len(chunk) < chunk_duration * 1000:
                chunk += AudioSegment.silent(duration=(chunk_duration * 1000) - len(chunk))  # Pad with silence
            # Export the chunk to an mp3 file
            chunk_filename = f"{str(i)}_chunk{j // (chunk_duration * 1000)}.mp3"
            chunk_filepath = os.path.join(download_path, chunk_filename)
            chunk.export(chunk_filepath, format='mp3')
        
        os.remove(temp_filepath)  # Remove the temporary file

def generate_spectrograms(audio_path='./raw_audio', spectrogram_path='./spectrograms'):
    # Ensure spectrogram directory exists
    if not os.path.exists(spectrogram_path):
        os.makedirs(spectrogram_path)

    # Parameters for STFT
    FRAME_SIZE = 2048
    HOP_SIZE = 512

    # Convert all audio chunks into spectrograms
    for audio_chunk in os.listdir(audio_path):
        if audio_chunk.endswith('.mp3'):
            audio_chunk_path = os.path.join(audio_path, audio_chunk)
            # Load audio file with librosa
            y, sr = librosa.load(audio_chunk_path)
            # Calculate the Short-Time Fourier Transform (STFT)
            S = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
            # Calculate the spectrogram
            Y = np.abs(S) ** 2
            # Convert the magnitude spectrogram to log scale
            # Ensure a minimum value for log scaling to avoid -inf which can occur due to padding silence
            Y_log = librosa.power_to_db(Y, ref=np.max, amin=1e-20)
            # Plot the spectrogram
            plt.figure(figsize=(25, 10))
            librosa.display.specshow(Y_log, sr=sr, hop_length=HOP_SIZE, x_axis="time", y_axis="log")
            plt.colorbar(format="%+2.f")
            plt.axis('off')  # No axis for a cleaner look
            spectrogram_filename = audio_chunk.replace('.mp3', '.png')
            spectrogram_filepath = os.path.join(spectrogram_path, spectrogram_filename)
            plt.savefig(spectrogram_filepath, bbox_inches='tight', pad_inches=0, dpi=300)  # Higher resolution
            plt.close()

import shutil

def separate_instruments(audio_path='./raw_audio', instrument_audio_path='./instrument_audio', batch_size=10):
    if not os.path.exists(instrument_audio_path):
        os.makedirs(instrument_audio_path)

    audio_chunks = [f for f in os.listdir(audio_path) if f.endswith(('.mp3', '.wav', '.ogg'))]
    audio_chunks = audio_chunks[:10]
    for i in range(0, len(audio_chunks), batch_size):
        batch_chunks = audio_chunks[i:i + batch_size]
        audio_chunk_paths = [os.path.join(audio_path, chunk) for chunk in batch_chunks]
        # Use spleeter to separate the audio chunks into stems in batch
        subprocess.call(['spleeter', 'separate', '-o', instrument_audio_path, '-p', 'spleeter:5stems', *audio_chunk_paths])

        # Move and rename the stems for each audio chunk in the batch
        for audio_chunk in batch_chunks:
            chunk_name = audio_chunk.split('.')[0]
            chunk_stem_path = os.path.join(instrument_audio_path, chunk_name)
            for stem in ['drums', 'bass', 'piano', 'other', 'vocals']:
                stem_filename = stem + '.wav'
                stem_path = os.path.join(chunk_stem_path, stem_filename)
                if os.path.exists(stem_path):
                    # Rename the file to include the original chunk name
                    new_stem_name = chunk_name + '_' + stem_filename
                    new_stem_path = os.path.join(instrument_audio_path, new_stem_name)
                    os.rename(stem_path, new_stem_path)

            # Remove the temporary output directory created by spleeter for this chunk
            if os.path.exists(chunk_stem_path):
                shutil.rmtree(chunk_stem_path)



def spectrogram_to_audio(spectrogram_path='./spectrograms', converted_audio_path='./converted_audio'):
    # Ensure converted audio directory exists
    if not os.path.exists(converted_audio_path):
        os.makedirs(converted_audio_path)

    # Parameters for inverse STFT
    frame_length = 2048
    frame_step = 512

    # Convert all spectrograms into audio files
    for spectrogram_filename in os.listdir(spectrogram_path):
        if spectrogram_filename.endswith('.png'):
            try:
                # Load the spectrogram image
                spectrogram_filepath = os.path.join(spectrogram_path, spectrogram_filename)
                with Image.open(spectrogram_filepath) as img:
                    spectrogram = np.array(img)

                # Convert the image data back to a spectrogram format
                # Inverse of the log scaling
                Y_log = librosa.db_to_power(spectrogram)

                # Estimate the phase using Griffin-Lim algorithm
                # librosa.griffinlim returns the time-domain audio signal reconstructed from the S matrix
                audio_reconstructed = librosa.griffinlim(Y_log, n_iter=32, hop_length=frame_step, win_length=frame_length)

                # Save the reconstructed audio
                audio_filename = spectrogram_filename.replace('.png', '.wav')
                audio_filepath = os.path.join(converted_audio_path, audio_filename)
                sf.write(audio_filepath, audio_reconstructed, 44100)  # Assuming a sample rate of 44100 Hz
            except Exception as e:
                print(f"Error processing {spectrogram_filename}: {e}")

#download_and_split_audio()
#generate_spectrograms()
separate_instruments()
#spectrogram_to_audio()