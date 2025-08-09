from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os, sys, importlib
import numpy as np
import random

class SeqDataset(Dataset):
    def __init__(self, device):
        self.data_instances = []
        self.device = device
    def init_feature_extractor(self, feature_extractor_file, feature_extractor_class_name):

        file_basename = os.path.basename(feature_extractor_file)
        mod_name = os.path.splitext(file_basename)[0]
        # print (file_basename, mod_name)

        spec = importlib.util.spec_from_file_location(mod_name, feature_extractor_file)
        feature_module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = feature_module

        spec.loader.exec_module(feature_module)
        Extractor = getattr(feature_module, feature_extractor_class_name)
        self.feature_extractor = Extractor()
        # print (self.feature_extractor)
        return

    def create_dataset_for_one_song(self, audio_path, song_id, do_svs):
        self.data_instances = []
        self.song_id = song_id
        feature_data = self.feature_extractor.get_feature_from_mixture(audio_path, do_svs=do_svs, device=self.device)

        print (feature_data.shape)

        frame_size = 1024.0 / 44100.0
        
        channel_num, frame_num, cqt_size = feature_data.shape[0], feature_data.shape[1], feature_data.shape[2]
        
        # To avoid cuda OOM
        num_frames = 20000
        frame_num = feature_data.shape[1]

        for frame_idx in range(0, frame_num, num_frames):
            start = frame_idx
            end = min(frame_num, frame_idx+num_frames)
            features = feature_data[:,start:end,:]
            self.data_instances.append(features)

    def __getitem__(self, idx):
        return (self.data_instances[idx], self.song_id)

    def __len__(self):
        return len(self.data_instances)
