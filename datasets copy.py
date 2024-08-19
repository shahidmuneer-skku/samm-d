import numpy as np
import os
from torch.utils.data import Dataset
import librosa
import torch
import torchaudio
import random
import cv2  # OpenCV for video processing
import torchvision
import traceback
import torch.nn.functional as F
from decord import VideoReader
from decord import cpu, gpu
import torchvision.transforms as transforms
import av
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

def pitch_shift(features, sr=16000, n_steps=2):
    shifted = librosa.effects.pitch_shift(features.numpy(), sr=sr, n_steps=n_steps)
    return torch.tensor(shifted)

def speed_change(features, rate=1.1):
    changed = librosa.effects.time_stretch(features.numpy(), rate=rate)
    return torch.tensor(changed)

def apply_augmentation(features):
    if random.random() < 0.5:
        features = add_noise(features)
    if random.random() < 0.5:
        features = mask_features(features)
    if random.random() < 0.5:
        features = time_shift(features)
    if random.random() < 0.5:
        features = pitch_shift(features)
    if random.random() < 0.5:
        features = speed_change(features)
    return features


def read_video_with_audio(path):
    container = av.open(path)
    video_frames = []
    audio_frames = []

    stream = container.streams.video[0]
    for frame in container.decode(stream):
        img = frame.to_image()  # Convert frame to PIL Image
        img_tensor = transforms.functional.to_tensor(img)
        video_frames.append(img_tensor)
        if len(video_frames) >= 40:
            break
    
    if container.streams.audio:
        audio_stream = container.streams.audio[0]
        for frame in container.decode(audio_stream):
            audio_data = frame.to_ndarray()
            audio_frames.append(audio_data)
    
    video_tensor = torch.stack(video_frames) if video_frames else None
    audio_tensor = np.concatenate(audio_frames, axis=1) if audio_frames else None

    return video_tensor, torch.tensor(audio_tensor, dtype=torch.float32) if audio_tensor is not None else None

def read_video_decord(path):
    vr = VideoReader(path, ctx=cpu(0))  # Use gpu(0) for GPU
    video = vr.get_batch(range(40)).asnumpy()  # Get first 40 frames
    video = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0  # Convert to torch tensor and normalize
    return video

def read_video(path: str):
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    video = video.permute(0, 3, 1, 2) / 255
    audio = audio.permute(1, 0)
    return video, audio, info

class SAMMD_dataset(Dataset):
    """
    Dataset class for the SAMMD2024 dataset.
    """
    def __init__(self, base_dir, partition="train", max_len=64000, frame_rate=1):
        assert partition in ["train", "dev", "test"], "Invalid partition. Must be one of ['train', 'dev', 'test']"
        self.base_dir = base_dir
        self.partition = partition
        self.max_len = max_len
        self.frame_rate = frame_rate  # Number of frames to extract per second
        self.real_dir = os.path.join(base_dir, f"{partition}/real")
        self.fake_dir = os.path.join(base_dir, f"{partition}/fake")
        
        self.file_list = []
        # Limit for samples from each category
        limit_per_category = 20
        
        # Process real videos
        count_real = 0
        print(self.real_dir)
        for root, _, files in os.walk(self.real_dir):
            for file in files:
                if file.endswith(("mp4", "avi", "mov")):
                    self.file_list.append((os.path.join(root, file), 1))  # 1 for real
                    count_real += 1
                    if count_real >= limit_per_category:
                        break
            if count_real >= limit_per_category:
                break
        
        count_fake = 0
        for root, _, files in os.walk(self.fake_dir):
            for file in files:
                if file.endswith(("mp4", "avi", "mov")):
                    self.file_list.append((os.path.join(root, file), 0))  # 0 for fake
                    count_fake += 1
                    if count_fake >= limit_per_category:
                        break
            if count_fake >= limit_per_category:
                break
        print(f"Total files are {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):            
        file_path, label = self.file_list[index]
        
        try:
            # Extract audio
            # audio_waveform, sample_rate = torchaudio.load(file_path)
            # audio_waveform = audio_waveform.mean(dim=0)  # Convert to mono
            # audio_waveform = librosa.resample(audio_waveform.numpy(), sample_rate, 16000)
            # audio_waveform = pad_random(audio_waveform, self.max_len)
            # audio_waveform = torch.tensor(audio_waveform)
            
            # if self.partition == "train":
            #     audio_waveform = apply_augmentation(audio_waveform)
            video, audio, info = read_video_decord(file_path)
            
            audio = pad_random(audio, self.max_len)
            audio = audio.permute(1,0)
            audio = audio.mean(dim=0)  # Convert to mono
            # audio = audio.unsqueeze(0)
            # Determine padding sizes for height and width
            pad_height = max(0, 1024 - video.shape[0])
            pad_width = max(0, 1024 - video.shape[0])
            
            # padding_size=(pad_width,pad_height)
            # video = F.pad(video, (0, 0, 0, 0, 0, 0, *padding_size))
            # if self.partition == "train":
            #     audio_waveform = apply_augmentation(audio_waveform)
            
    # Adjust video to have exactly 40 frames
            # num_frames = video.shape[0]
            # if num_frames < 1024:
            #     pad_size = 40 - num_frames  # Calculate how many frames are missing
            #     padding = torch.zeros(pad_size, *video.shape[1:], device=video.device)  # Create a padding tensor
            #     video = torch.cat([video, padding], dim=0)  # Concatenate the video and padding tensor along the time dimension

            video = video[:40]
            # print(audio.shape,video.shape)
            text = "I AM Text"
            # print(audio.shape)
            return audio, video,text, label, os.path.basename(file_path)
        except Exception as e:
            traceback.print_exc()
            print(f"Error loading {file_path}: {e}")
            return self.__getitem__((index + 1) % len(self.file_list))
