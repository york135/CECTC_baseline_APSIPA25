import librosa
import numpy as np
import soundfile as sf
import torch
import sys, os
from tqdm import tqdm
import random
import shutil, os

if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    training_dir = os.path.join(dataset_dir, 'JBM555_train')
    validation_dir = os.path.join(dataset_dir, 'JBM555_validation')
    test_dir = os.path.join(dataset_dir, 'JBM555_test')

    train_list = 'list_ID_JBM3000_train.txt'
    validation_list = 'list_ID_JBM3000_validation.txt'
    test_list = 'list_ID_JBM3000_test.txt'

    train_song_data = np.loadtxt(train_list, dtype=str)
    validation_song_data = np.loadtxt(validation_list, dtype=str)
    test_song_data = np.loadtxt(test_list, dtype=str)

    if not os.path.exists(training_dir):
        os.mkdir(training_dir)

    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    for audio_name in tqdm(os.listdir(dataset_dir)):
        src_dir = os.path.join(dataset_dir, audio_name)

        if audio_name in train_song_data:
            dst_dir = os.path.join(training_dir, audio_name)
        elif audio_name in validation_song_data:
            dst_dir = os.path.join(validation_dir, audio_name)
        elif audio_name in test_song_data:
            dst_dir = os.path.join(test_dir, audio_name)
        else:
            print ('???', audio_name)
            continue
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

