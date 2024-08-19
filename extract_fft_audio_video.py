import torch
import torchvision
import numpy as np
import cv2
import os
import librosa
def read_video(path: str):
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    video = video.permute(0, 3, 1, 2) / 255  # Normalize video frames
    audio = audio.permute(1, 0)  # Rearrange audio for processing
    return video, audio, info

def fft_to_image(f):
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Log scale, add 1 to avoid log(0)
    magnitude_image = np.uint8(magnitude_spectrum / magnitude_spectrum.max() * 255)
    return magnitude_image

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

def process_video_audio_fft(video, audio):
    # Define the dimensions for the output video, e.g., 224x224 for simplicity
    out_height, out_width = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter('output_fft_video.mp4', fourcc, 25.0, (out_width, out_height))

    for i in range(len(video)):
        frame = video[i].numpy().transpose(1, 2, 0)  # Convert to HxWxC
        gray_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray_frame)
        magnitude_image, ifft_image = fft_to_image_and_ifft(f)
        magnitude_image = cv2.resize(ifft_image, (out_width, out_height))
        color_mapped_image = cv2.applyColorMap(magnitude_image, cv2.COLORMAP_JET)
        out.write(color_mapped_image)

    # Audio FFT processing
    # audio_np = audio.numpy()
    # diagnose_audio(audio_np)
    # out = cv2.VideoWriter('output_fft_audio.mp4', fourcc, 25.0, (out_width, out_height))
    # window_size = 1024  # Size of the window to apply FFT on
    # hop_length = window_size // 2  # 50% overlap

    # # We assume audio has two channels; process each channel separately
    # if audio_np.shape[0] == 1:  # Stereo audio
    #     audio_channel = audio_np[0]  # Extract the single audio channel

    #     for start in range(0, len(audio_channel) - window_size, hop_length):
    #         # Compute Mel Spectrogram
    #         S = librosa.feature.melspectrogram(y=audio_channel[start:start + window_size], sr=44100, n_fft=window_size, hop_length=hop_length, n_mels=128)
    #         S_dB = librosa.power_to_db(S, ref=np.max)

    #         # Compute MFCC
    #         mfccs = librosa.feature.mfcc(S=S_dB, n_mfcc=13)
            
    #         # Optionally, apply FFT to MFCCs
    #         mfccs_fft = np.fft.fft(mfccs, axis=1)
    #         mfccs_fft_shift = np.fft.fftshift(mfccs_fft, axes=1)
    #         magnitude_mfcc_fft = 20 * np.log10(np.abs(mfccs_fft_shift) + 1e-6)  # Log scale

    #         # Map the magnitude to an image
    #         magnitude_image = np.uint8(255 * (magnitude_mfcc_fft - magnitude_mfcc_fft.min()) / (magnitude_mfcc_fft.max() - magnitude_mfcc_fft.min()))
    #         color_mapped_image = cv2.applyColorMap(magnitude_image, cv2.COLORMAP_JET)
    #         color_mapped_image = cv2.resize(color_mapped_image, (out_width, out_height))
    #         out.write(color_mapped_image)
    out.release()  # Release the video writer

def main():
    base_dir = "/media/data1/FakeAVCeleb/"
    partition = "train/fake"
    video_path = os.path.join(base_dir, f"{partition}/FakeVideo-FakeAudio-id00018-00181_id00243_oWwEIXM3oZQ_id00945_wavtolip.mp4")
    video, audio, info = read_video(video_path)
    audio = audio.permute(1,0)
    print(audio.shape)
    process_video_audio_fft(video, audio)

if __name__ == "__main__":
    main()
