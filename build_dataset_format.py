import librosa
import numpy as np
import soundfile as sf
import torch
import sys, os
from tqdm import tqdm
import random


if __name__ == "__main__":
    audio_dir = sys.argv[1]
    separated_dir = sys.argv[2]
    output_dir = sys.argv[3]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for audio_name in tqdm(os.listdir(audio_dir)):
        mix_path = os.path.join(audio_dir, audio_name)
        y_mix, _ = librosa.load(mix_path, sr=44100, mono=True)

        voc_path = os.path.join(separated_dir, audio_name)
        y_voc, _ = librosa.load(voc_path, sr=44100, mono=True)

        output_song_dir = os.path.join(output_dir, audio_name[:-4])
        if not os.path.exists(output_song_dir):
            os.mkdir(output_song_dir)

        output_voc_path = os.path.join(output_song_dir, 'Vocal.wav')
        sf.write(
            output_voc_path,
            y_voc,
            44100,
            "PCM_16",
        )

        y_acc = y_mix - y_voc
        output_acc_path = os.path.join(output_song_dir, 'Inst.wav')
        sf.write(
            output_acc_path,
            y_acc,
            44100,
            "PCM_16",
        )