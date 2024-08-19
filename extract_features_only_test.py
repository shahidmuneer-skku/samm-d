import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from urllib.request import urlopen
import os 
import cv2
import pandas as pd
import numpy as np
import librosa
import torch.nn as nn
def read_video(path: str):
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    video = video.permute(0, 3, 1, 2) / 255
    audio = audio.permute(1, 0)
    return video, audio, info

base_dir = "/media/data1/FakeAVCeleb/"
# partition = "train"
# max_len = 256
# frame_rate = 25  # Number of frames to extract per second
# real_dir = os.path.join(base_dir, f"{partition}/realOriginal")
# fake_dir = os.path.join(base_dir, f"{partition}/fake")

# file_list = []
# print(real_dir)
# for root, _, files in os.walk(real_dir):
#     for file in files:
#         if file.endswith(("mp4", "avi", "mov")):
#             file_list.append({"path":os.path.join(root, file), "label":1})  # 1 for real
           

# for root, _, files in os.walk(fake_dir):
#     for file in files:
#         if file.endswith(("mp4", "avi", "mov")):
#             file_list.append({"path":os.path.join(root, file), "label":0})  # 0 for fake
           
# print(f"Total files are in {partition} are {len(file_list)}")


# Load the model and tokenizer from Hugging Face Hub
device = "cuda"
model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14', device=device)
tokenizer = get_tokenizer('ViT-H-14')
# Helper function to process video frames
def process_video_frames(video):
    # frames = video.permute(0, 3, 1, 2) / 255.0  # Normalize frames
    frames = video 
    frame_embeddings = []
    for frame in frames:
        frame = frame.permute(1,2,0)
        frame = frame.numpy().astype('uint8')
        # print(frame.shape)
        frame_image = Image.fromarray(frame)
        processed_frame = preprocess(frame_image).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            processed_frame= processed_frame.to(device)
            frame_features = model.encode_image(processed_frame)
            frame_features = F.normalize(frame_features, dim=-1)
        frame_embeddings.append(frame_features.cpu().numpy())
    return frame_embeddings

def fft_to_image(f):
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Log scale, add 1 to avoid log(0)
    magnitude_image = np.uint8(magnitude_spectrum / magnitude_spectrum.max() * 255)
    return magnitude_image

def diagnose_audio(audio_np):
    print("Audio Shape:", audio_np.shape)
    print("Max Value in Audio:", np.max(audio_np))
    print("Min Value in Audio:", np.min(audio_np))
    # Assuming stereo audio
    if audio_np.shape[0] == 2:
        print("Stereo audio confirmed")
        for i, channel in enumerate(['Left', 'Right']):
            max_val = np.max(audio_np[i])
            min_val = np.min(audio_np[i])
            print(f"{channel} channel: Max={max_val}, Min={min_val}")
            if max_val == 0 and min_val == 0:
                print(f"Warning: {channel} channel is silent!")

def process_video_audio_fft(video, audio, output_dir):
    # Define the dimensions for the output video, e.g., 224x224 for simplicity
    out_height, out_width = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter(f'{output_dir}_video_fft.mp4', fourcc, 25.0, (out_width, out_height))
    for i in range(len(video)):
        frame = video[i].numpy().transpose(1, 2, 0)  # Convert to HxWxC
        gray_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray_frame)
        magnitude_image = fft_to_image(f)
        magnitude_image = cv2.resize(magnitude_image, (out_width, out_height))
        color_mapped_image = cv2.applyColorMap(magnitude_image, cv2.COLORMAP_JET)
        out.write(color_mapped_image)
    # Audio FFT processing
    audio_np = audio.numpy()
    # diagnose_audio(audio_np)
    out = cv2.VideoWriter(f'{output_dir}_audio_fft.mp4', fourcc, 25.0, (out_width, out_height))
    window_size = 1024  # Size of the window to apply FFT on
    hop_length = window_size // 2  # 50% overlap
    # We assume audio has two channels; process each channel separately
    if audio_np.shape[0] == 1:  # Stereo audio
        audio_channel = audio_np[0]  # Extract the single audio channel
        for start in range(0, len(audio_channel) - window_size, hop_length):
            # Compute Mel Spectrogram
            S = librosa.feature.melspectrogram(y=audio_channel[start:start + window_size], sr=44100, n_fft=window_size, hop_length=hop_length, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            # Compute MFCC
            mfccs = librosa.feature.mfcc(S=S_dB, n_mfcc=13)
            # Optionally, apply FFT to MFCCs
            mfccs_fft = np.fft.fft(mfccs, axis=1)
            mfccs_fft_shift = np.fft.fftshift(mfccs_fft, axes=1)
            magnitude_mfcc_fft = 20 * np.log10(np.abs(mfccs_fft_shift) + 1e-6)  # Log scale
            # Map the magnitude to an image
            magnitude_image = np.uint8(255 * (magnitude_mfcc_fft - magnitude_mfcc_fft.min()) / (magnitude_mfcc_fft.max() - magnitude_mfcc_fft.min()))
            color_mapped_image = cv2.applyColorMap(magnitude_image, cv2.COLORMAP_JET)
            color_mapped_image = cv2.resize(color_mapped_image, (out_width, out_height))
            out.write(color_mapped_image)
    out.release()  # Release the video writer

# Example text and video path
def save_features( output_dir,row):    
    labels_list = [f"These video and audio frames are of {row['race']} {row['gender']}, which is {row['method']} and it has {row['type']}, your job is to learn the patterns of the audio and video. "]
    # Tokenize tex
    print(labels_list[0])
    text = tokenizer(labels_list, context_length=model.context_length)
    # Process the video and generate frame embeddings
    # frame_embeddings = process_video_frames(video)
    # Convert list of arrays into a single tensor
    # frame_embeddings_tensor = torch.tensor(np.vstack(frame_embeddings))
    with torch.no_grad(), torch.cuda.amp.autocast():
        text = text.to(device)
        text_features = model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
    # frame_embeddings_tensor = frame_embeddings_tensor.cpu()
    text_features = text_features.cpu()
    # np.save(f'{output_dir}_video_embds.npy', frame_embeddings_tensor)
    np.save(f'{output_dir}_text_embds_row.npy', text_features)

    # video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
def save_ffts(video, audio, output_dir):
    audio = audio.permute(1,0)
    process_video_audio_fft(video, audio, output_dir)

def save(video_path,output_dir, row):    
    # video, audio, info = read_video(video_path)
    # print(video.shape)
    # save_ffts(video, audio,output_dir)
    save_features(output_dir,row)




def main():
    base_dir = "/home/PromptCLIP/FakeAVCeleb_v1.2/"
    csv_path = f'{base_dir}meta_data.csv'  # Change this to the actual path of your CSV file
    data = pd.read_csv(csv_path)
    # data = data[1348*15:1348*16]
    for index, row in data.iterrows():
        path = os.path.join(base_dir,row['Unnamed: 9'].replace("FakeAVCeleb/", ""))
        video_path = os.path.join(path, row["path"])
        print(video_path)
        output_dir = os.path.join(path, row["path"].replace(".mp4", ""))
        save(video_path, output_dir,row)
if __name__ == "__main__":
    main()
