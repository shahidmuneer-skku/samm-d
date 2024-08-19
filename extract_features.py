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
import time
import torch.multiprocessing as mp
from transformers import CLIPProcessor, CLIPModel
import concurrent.futures
import threading
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
# model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14', device=device)
# tokenizer = get_tokenizer('ViT-H-14')
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

def fft_to_image_and_ifft(f):
    fshift = np.fft.fftshift(f)
    
    # Convert FFT to a visual representation
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Log scale, add 1 to avoid log(0)
    magnitude_image = np.uint8(magnitude_spectrum / magnitude_spectrum.max() * 255)
    
    # Apply Inverse FFT (IFFT)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize the IFFT output to 0-255 and convert to uint8 for visualization
    img_back_normalized = np.uint8(img_back / np.max(img_back) * 255)
    
    return magnitude_image, img_back_normalized
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

    out = cv2.VideoWriter(f'{output_dir}_video_ifft.mp4', fourcc, 25.0, (out_width, out_height))
    for i in range(len(video)):
        frame = video[i].numpy().transpose(1, 2, 0)  # Convert to HxWxC
        gray_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray_frame)
        magnitude_image, ifft_image = fft_to_image_and_ifft(f)
        
        magnitude_image = cv2.resize(ifft_image, (out_width, out_height))
        color_mapped_image = cv2.applyColorMap(magnitude_image, cv2.COLORMAP_JET)
        out.write(color_mapped_image)
    # Audio FFT processing
    audio_np = audio.numpy()
    # diagnose_audio(audio_np)
    out = cv2.VideoWriter(f'{output_dir}_audio_fft.mp4', fourcc, 25.0, (out_width, out_height))
    window_size = 1024  # Size of the window to apply FFT on
    hop_length = window_size // 2  # 50% overlap
    # # We assume audio has two channels; process each channel separately
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
def save_features(video, output_dir,row, model, processor):    
    audio_type = "Fake Audio" if row["modify_audio"]==True else "Real Audio"  
    video_type = "Fake Video" if row["modify_video"]==True else "Real Video"  
    labels_list = [f"Learn the pattern of the video and audio containing {video_type}, and {audio_type}"]
    
    # print(save_features)#  Tokenize tex
    # text = tokenizer(labels_list, context_length=model.context_length)
    # Process the video and generate frame embeddings
    # frame_embeddings = process_video_frames(video)
    # # Convert list of arrays into a single tensor
    # frame_embeddings_tensor = torch.tensor(np.vstack(frame_embeddings))
    # with torch.no_grad(), torch.cuda.amp.autocast():
    #     text = text.to(device)
    #     text_features = model.encode_text(text)
    #     text_features = F.normalize(text_features, dim=-1)
    #     text_features = F.normalize(text_features, dim=-1)
    # frame_embeddings_tensor = frame_embeddings_tensor.cpu()
    # text_features = text_features.cpu()
    frame_embeddings = []
    text_embeddings = []
    # for frame in video:
    #     frame = frame.permute(1,2,0)
    #     frame = frame.numpy().astype('uint8')
    #     # print(frame.shape)
    #     frame_image = Image.fromarray(frame)
    video = torch.nn.functional.interpolate(video, size=(244, 244), mode='bilinear', align_corners=False)

    inputs = processor(text=labels_list, images=video, return_tensors="pt", padding=True, do_rescale=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    
    np.save(f'{output_dir}_video_embds_clip_vit.npy', outputs.image_embeds.cpu().detach().numpy())
    np.save(f'{output_dir}_text_embds_clip_vit.npy', outputs.text_embeds.cpu().detach().numpy())

    # video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
def save_ffts(video, audio, output_dir):
    audio = audio.permute(1,0)
    process_video_audio_fft(video, audio, output_dir)

def save(video_path,output_dir,row,model, processor):    
    video, audio, info = read_video(video_path)
    # print(video.shape)
    save_ffts(video, audio,output_dir)
    save_features(video,output_dir,row, model, processor)

def extractFeatures(chunk, model_name, device):
    
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)#.to("cuda:3")
    model = model.to(device)
    # print(f"chunk 2 is reached {len(chunk)}")
    
    base_dir = "/media/NAS/DATASET/LAV-DF/LAV-DF/"
    # model = None
    # processor= None
    for index, row in chunk.iterrows():
        # path = os.path.join(base_dir,row['Unnamed: 9'].replace("FakeAVCeleb/", ""))
        video_path = row["video_path"].replace("_video_fft.mp4",".mp4")
        output_dir = row["video_path"].replace(".mp4","")
        # time.sleep(1)
        # print(video_path)
        if os.path.exists(video_path):
            print(f"Process id is  {video_path}")
            save(video_path, output_dir,row,model, processor)

def main():
    # base_dir = "/media/NAS/DATASET/LAV-DF/LAV-DF/"
    # base_dir = "/media/NAS/DATASET/DFDC-Official/full_set/test"
    # csv_path = f'{base_dir}metadata.json'  # Change this to the actual path of your CSV file
    # data = pd.read_json(csv_path)
    # data = data[1348*15:1348*16]
    # Correct the paths in the DataFrame
    # data['full_path'] = data['Unnamed: 9'].str.replace("FakeAVCeleb/", "")
    # data['video_path'] = data.apply(lambda row: os.path.join(base_dir, row['full_path'], f"{row['path'].replace('.mp4','')}_video_fft.mp4"), axis=1)
    # data = data[data["split"]=="train"]
    # data["video_path"] =  data["file"].apply(lambda x: os.path.join(base_dir, x.replace(".mp4", "_video_fft.mp4")))
    # data['file_exists'] = data['video_path'].apply(os.path.exists)
    # non_existing_videos  = data[~data['file_exists']]
  
    chunk_size = 16995
    chunks = [data[i:i+chunk_size] for i in range(0, len(non_existing_videos), chunk_size)]
    print(len(chunks))
    model_name = "openai/clip-vit-base-patch32"
           
    # Assign each chunk to a different GPU if available
    num_gpus = torch.cuda.device_count()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = []
    #     for chunk in chunks:
    #         # # gpu_id = 3 if gpu_id == 2 else gpu_id
    #         # device = f'cuda:{gpu_id}'  # Round-robin assignment
    #         # model = model.to(device)
    #         device = ""
    processes = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = []
        for i,chunk in enumerate(chunks):  # Add enumerate to get index 'i'
          device = f'cuda:{i % num_gpus}'
        #   print(chunk)
          futures.append(executor.submit(extractFeatures, chunk, model_name, device))
        for future in concurrent.futures.as_completed(futures):
                future.result()
if __name__ == "__main__":
    mp.set_start_method('spawn')  # For CUDA multiprocessing
    main()