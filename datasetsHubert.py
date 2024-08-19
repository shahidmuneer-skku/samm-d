import numpy as np
import os
from torch.utils.data import Dataset
import librosa
import torch
import torchaudio 
import random
def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))
    return pad_random(padded_x, max_len)

def add_noise(features, noise_level=0.005):
    noise = torch.randn(features.size()) * noise_level
    return features + noise

def mask_features(features, mask_prob=0.1, mask_value=0.0):
    mask = torch.rand(features.size()) < mask_prob
    features = features.masked_fill(mask, mask_value)
    return features

def time_shift(features, shift_limit=0.2):
    shift = int(random.uniform(-shift_limit, shift_limit) * features.size(1))
    return torch.roll(features, shifts=shift, dims=1)

def apply_augmentation(features):
    if random.random() < 0.5:
        features = add_noise(features)
    if random.random() < 0.5:
        features = mask_features(features)
    if random.random() < 0.5:
        features = time_shift(features)
    return features

class SAMMD_dataset(Dataset):
    """
    Dataset class for the SAMMD2024 dataset.
    """
    def __init__(self, base_dir, partition="train", max_len=64000):
        assert partition in ["train", "dev", "test"], "Invalid partition. Must be one of ['train', 'dev', 'test']"
        self.base_dir = base_dir
        self.partition = partition
        self.base_dir = os.path.join(base_dir, partition + "_set")
        self.max_len = max_len
        self.file_list= []
        try:
            with open(os.path.join(base_dir, f"{partition}.txt"), "r") as f:
                filelists = f.readlines()
                for file in filelists:
                    file_name = file.split(" ")[2].strip()
                    bonafide_or_spoof = file.split(" ")[-1].strip()
                    set_ = "train_set" if partition=="train" else "dev_set"
                    # print(os.path.join(os.path.join(base_dir,set_),f"{file_name}.flac"))
                    if(os.path.exists(os.path.join(os.path.join(base_dir,set_),f"{file_name}.flac.npy"))):
                        self.file_list.append(file)
                
                for root, _, files in os.walk(os.path.join(os.path.dirname(self.base_dir), 'results')):
                    for file in files:
                        # print(file)
                        if file.endswith("_hu.npy"):
                            self.file_list.append(file)
                # if len(self.file_list==0):
                #     assert 0, f"self.file_list is empty after processing partition: {partition}"
        except Exception as e:
            if partition == "test":
                self.file_list = []
                # get all *.flac files in the test_set directory
                for root, _, files in os.walk(self.base_dir):
                    for file in files:
                        # print(file)
                        # if file.endswith(".flac.npy"):
                        if file.endswith(".flac"):

                            self.file_list.append(file)
            else:
                raise FileNotFoundError(f"File {partition}.txt not found in {base_dir}")
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):            
        if self.partition == "test":
            file_name = self.file_list[index].strip()
            path = os.path.join(self.base_dir, file_name)
            label = 0 # dummy label. Not used for test set.
        else:
            file = self.file_list[index]
            # print(file)
            if len(file.split(" "))<2:
                file_name = os.path.join("results", os.path.basename(file))
                path = os.path.join(os.path.dirname(self.base_dir), file_name)
                # print(file_name)
                # file_name = file_name+".npy"
                label = 0
            else:
                file_name = file.split(" ")[2].strip()
                file_name = file_name+".flac.npy"
                bonafide_or_spoof = file.split(" ")[-1].strip()
                label = 1 if bonafide_or_spoof == "bonafide" else 0
                path = os.path.join(self.base_dir, file_name)
        try:
            # print(file_name)
            # x, _ = librosa.load(os.path.join(self.base_dir, file_name + ".flac"), sr=16000, mono=True)
            # x, _ = librosa.load(path, sr=16000, mono=True)
            # print(os.path.join(self.base_dir, file_name+".flac.npy"))
            
            ##### Numpy files
            x = np.load(path)
            # print(x.shape)
            x = torch.from_numpy(x)
            # x = pad_random(x, 512)
            # x= x.permute(1,0)
            x = x[:, 0, :]
            padding_right = max(0, 1528 - x.shape[0])
            x = x.permute(1,0)
            if padding_right > 0:
                x = torch.nn.functional.pad(x, (0, padding_right))
            # x = librosa.util.normalize(x)
            # if self.partition == "train":
            #     x = apply_augmentation(x)
            # x = torch.flatten(x,start_dim=1,end_dim=2)
            # x = x.permute(1,0)
            # print(x.shape)
            x = x.permute(1,0)
            #### numpy files

            # file_name is used for generating the score file for submission
            return x, label, file_name
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return None
            # return self.__getitem__((index + 1) % len(self.file_list))