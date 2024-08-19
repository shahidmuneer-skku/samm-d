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
import pandas as pd
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
    # del 
    return video_tensor, torch.tensor(audio_tensor, dtype=torch.float32) if audio_tensor is not None else None

def read_video_decord(path):
    vr = VideoReader(path, ctx=cpu(0))  # Use gpu(0) for GPU
    video = vr.get_batch(range(40)).asnumpy()  # Get first 40 frames
    video = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0  # Convert to torch tensor and normalize
    return video

def extract_audio_with_ffmpeg(path, output_format='wav', sample_rate=16000):
    # Temporary output file
    temp_audio = 'temp_audio.wav'
    # Command to extract audio
    command = ['ffmpeg', '-i', path, '-ar', str(sample_rate), '-ac', '1', temp_audio, '-y']
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Load the processed audio file
    audio, sr = librosa.load(temp_audio, sr=sample_rate)
    audio_tensor = torch.tensor(audio).float()

    # Delete the temporary audio file after reading
    os.remove(temp_audio)

    return audio_tensor

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
        self.real_dir = os.path.join(base_dir, f"{partition}/realOriginal")
        self.fake_dir = os.path.join(base_dir, f"{partition}/fake")
        
        self.file_list = []
        # Limit for samples from each category
        limit_per_category = 64
        # base_dir = "/home/shahid/PromptCLIP/FakeAVCeleb_v1.2/"
        csv_path = f'{self.base_dir}meta_data.csv'  # Change this to the actual path of your CSV file
        data = pd.read_csv(csv_path)
        # Process real videos
        count_real = 0
        print(self.real_dir)
        for index, row in data.iterrows():
            path = os.path.join(base_dir,row['Unnamed: 9'].replace("FakeAVCeleb/", ""))
            video_path = os.path.join(path, row["path"])
            audio_fft = os.path.join(path, row["path"].replace(".mp4", "_audio_fft.mp4"))
            video_fft = os.path.join(path, row["path"].replace(".mp4", "_video_fft.mp4"))
            text_embeddings = os.path.join(path, row["path"].replace(".mp4", "_text_embds_clip_vit.npy"))
            video_embeddings = os.path.join(path, row["path"].replace(".mp4","_video_embds_clip_vit.npy"))
            label = 0 if row["method"]=="real" else 1 
            # if label == 0:
            #     print(label)   
            if os.path.exists(video_fft) and os.path.exists(video_path) and os.path.exists(video_embeddings):
                self.file_list.append({
                                        "video_path":video_path, 
                                        "audio_fft":audio_fft,
                                        "video_fft":video_fft,
                                        "text_embeddings":text_embeddings,
                                        "video_embeddings":video_embeddings,   
                                        "label":label})  # 1 for real
                   
        # for root, _, files in os.walk(self.real_dir):
        #     for file in files:
        #         if file.endswith(("mp4", "avi", "mov")):
        #             self.file_list.append({"path":os.path.join(root, file), "label":1})  # 1 for real
        #             count_real += 1
        #     #         if count_real >= limit_per_category:
        #     #             break
        #     # if count_real >= limit_per_category:
        #     #     break
        
        # count_fake = 0
        # for root, _, files in os.walk(self.fake_dir):
        #     for file in files:
        #         if file.endswith(("mp4", "avi", "mov")):
        #             self.file_list.append({"path":os.path.join(root, file), "label":0})  # 0 for fake
        #             count_fake += 1
            #         if count_fake >= limit_per_category:
            #             break
            # if count_fake >= limit_per_category:
            #     break
        print(f"Total files are in {partition} are {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):            
        
        try:
            data = self.file_list[index]
            file_path = data["video_path"]
            audio_fft = data["audio_fft"]
            video_fft = data["video_fft"]
            text_embeddings = data["text_embeddings"]
            video_embeddings = data["video_embeddings"]
            label = data["label"]
            # print(file_path)
            # if file_path == "/media/data2/FakeAVCeleb_v1.2/FakeAVCeleb/train/real/RealVideo-RealAudio-id00350-00015.mp4":
            #     self.__getitem__(index+1)
            # Extract audio
            # audio_waveform, sample_rate = torchaudio.load(file_path)
            # audio_waveform = audio_waveform.mean(dim=0)  # Convert to mono
            # audio_waveform = librosa.resample(audio_waveform.numpy(), sample_rate, 16000)
            # audio_waveform = pad_random(audio_waveform, self.max_len)
            # audio_waveform = torch.tensor(audio_waveform)
            
            # if self.partition == "train":
            #     audio_waveform = apply_augmentation(audio_waveform)
            # print(file_path)
            # edia/data2/FakeAVCeleb_v1.2/FakeAVCeleb/train/real/RealVideo-RealAudio-id00857-00347.mp4
            video, audio, info = read_video(file_path)
            
            # audio_fft,_,_ = read_video(audio_fft)
            video_fft,_,_ = read_video(video_fft)
            text_embeddings = np.load(text_embeddings)
            video_embeddings = np.load(video_embeddings)
            audio = audio.permute(1,0)
            audio = audio.mean(dim=0)  # Convert to mono
            audio = pad_random(audio, self.max_len)
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)

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
            # print(video_embeddings.shape)
            video = video[:40]
            video_embeddings = video_embeddings[:40]
            video_fft = video_fft[:40]
            # audio_fft = audio_fft[:40]
            # video = video[:40]# Assuming video is your tensor with the shape (40, 3, 244, 244)
            video_shape = video.shape[0]  # current number of frames, 40 in your case
            target_frames = 512  # target number of frames
            # print(video.shape)
            video = torch.nn.functional.interpolate(video, size=(244, 244), mode='bilinear', align_corners=False)
            # audio_fft = torch.nn.functional.interpolate(audio_fft, size=(244, 244), mode='bilinear', align_corners=False)
            video_fft = torch.nn.functional.interpolate(video_fft, size=(244, 244), mode='bilinear', align_corners=False)
            # assert audio_fft.shape[0] > 0, "audio_fft has zero frames after interpolation, which is unexpected."
            audio_fft = torch.randn(0,123)
            
            if video_fft.shape[0]<40 and video_embeddings.shape[0]<40:
                print(video_fft.shape[0],audio_fft.shape[0], video_embeddings.shape[0])
                return self.__getitem__((index + 1) % len(self.file_list))

            

            # print(video.shape,video_fft.shape, audio_fft.shape)
            return audio, video, label, audio_fft, video_fft, text_embeddings, video_embeddings, os.path.basename(file_path)
        except Exception as e:
            # traceback.print_exc()
            print(f"Error loading {file_path}: {e}")
            return self.__getitem__((index + 1) % len(self.file_list))
