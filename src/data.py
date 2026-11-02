import os
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrogram(audio_path, sr=32000, n_mels=128, fmax=14000):
    """
    Loads an audio file and plots its Mel-spectrogram.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=fmax, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram: {audio_path.split("/")[-1]}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

class BirdCLEFAudioDataset(Dataset):
    def __init__(self, df, audio_dir, target_sr=32000, duration=5):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.target_length = target_sr * duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        
        try:
            # Load audio using torchaudio
            waveform, sr = torchaudio.load(file_path)
            
            # Convert stereo to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample if sample rate doesn't match
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
                
            # Pad or truncate to ensure uniform length (1, target_length)
            if waveform.shape[1] > self.target_length:
                start = torch.randint(0, waveform.shape[1] - self.target_length, (1,)).item()
                waveform = waveform[:, start:start + self.target_length]
            else:
                padding = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
            label = torch.tensor(row['label_id'], dtype=torch.long)
            return waveform, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Fallback tensor
            return torch.zeros((1, self.target_length), dtype=torch.float32), torch.tensor(row['label_id'], dtype=torch.long)