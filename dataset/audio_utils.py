import os

import matplotlib.pyplot as plt

from SpecGAN.dataset.hparams import *
from scipy import signal

import librosa
import glob
import numpy as np
import subprocess
import librosa.display
from librosa.filters import mel


def normalize(m_spec):
    return np.clip((m_spec - MIN_LEVEL_DB) / -float(MIN_LEVEL_DB), 0, 1)


def denormalize(m_spec):
    return (np.clip(m_spec, 0, 1) * -MIN_LEVEL_DB) + MIN_LEVEL_DB


# custom STFT
class MelSpec:
    def __init__(self,
                 mel_f_min=0.0,
                 mel_f_max=None,
                 power=0):
        self.window = signal.get_window(window=WINDOW, Nx=WIN_LENGTH)
        self.mel_basis = mel(sr=SAMPLING_RATE,
                             n_fft=N_FFT,
                             n_mels=N_MEL_CHANNELS,
                             fmin=mel_f_min,
                             fmax=mel_f_max)
        self.power = power

    def __call__(self, audio: np.ndarray, norm=True):
        x = audio.copy()
        fft = librosa.stft(y=audio,
                           n_fft=N_FFT,
                           hop_length=HOP_SIZE,
                           win_length=WIN_LENGTH,
                           window=WINDOW,
                           center=True)
        if self.power:
            magnitude = np.abs(fft) ** self.power
        else:
            magnitude = np.abs(fft)

        mel_spec = np.matmul(self.mel_basis, magnitude)

        mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

        if norm:
            mel_spec = normalize(mel_spec)

        return mel_spec


def convert_mid_to_wav(filenames: list, folder_to_save='wave_gan_dataset'):
    os.makedirs(folder_to_save, exist_ok=True)
    for i, file in enumerate(filenames):
        print("Converting...")
        subprocess.call(['timidity', file, '-Ow', '-o', 'wave_gan_dataset/audio_{:d}.wav'.format(i)])


def convert_mp3_to_wav(filenames: list, folder_to_save: str = 'wav_files'):
    os.makedirs(folder_to_save, exist_ok=True)
    for i, file in enumerate(filenames):
        subprocess.call(['ffmpeg', '-i', file, '-acodec', 'pcm_u8', '-ar', '16000',
                         'wav_files/audio_{:d}.wav'.format(i)])


if __name__ == "__main__":
    audio, sr = librosa.load('wave_gan_dataset/audio_5.wav')
    print(audio.shape[0] / sr)
