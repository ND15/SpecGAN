import glob

import matplotlib.pyplot as plt
import soundfile
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
from SpecGAN.dataset.audio_utils import MelSpec, normalize, denormalize
from SpecGAN.dataset.hparams import *

make_mel = MelSpec()


def prepare_audio(audio):
    # 50 secs of audio
    audio = audio[:-1]

    mel_spec = make_mel(audio)

    mean = np.mean(mel_spec, axis=1)
    std = np.std(mel_spec, axis=1)
    return audio[..., np.newaxis], mel_spec[:, :128, np.newaxis]


def prepare_dataset(folder_path):
    filenames = glob.glob(folder_path + '/*.wav')[:1000]
    audios, mels = [], []

    for i in filenames:
        audio, sr = librosa.load(i, sr=SAMPLING_RATE)
        print("Processing file ", i)
        x = prepare_audio(audio)
        audios.append(x[0])
        mels.append(x[1])

    audio_dataset = tf.data.Dataset.from_tensor_slices((mels, audios))
    audio_dataset = audio_dataset.shuffle(buffer_size=BUFFER_SIZE)
    # audio_dataset = audio_dataset.repeat()
    audio_dataset = audio_dataset.batch(BATCH_SIZE, drop_remainder=True)
    audio_dataset = audio_dataset.prefetch(buffer_size=AUTOTUNE)
    return audio_dataset


if __name__ == "__main__":
    dataset = prepare_dataset('/home/nikhil/Downloads/archive/nsynth-train-all/audio')
    for x, y in dataset:
        print(x.shape, y.shape)
