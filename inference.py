import argparse
import os
import torch
import torch.nn as nn
from models.model_fft_128 import SAMMDModel
from util import set_seed
import cv2
import torchvision
import numpy as np

from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"
def read_video(path: str):
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    video = video[:128]
    video = video.permute(0, 3, 1, 2) / 255
    audio = audio.permute(1, 0)
    return video, audio, info
def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))


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

def single_video_inference(args, video_path):
    # Set the seed for reproducibility
    set_seed(args.random_seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = SAMMDModel(frontend=args.encoder, device=device).to(device)
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Load your video data here
    # For demonstration, I'm assuming 'video' is already a preprocessed tensor\
    video, audio, info = read_video(video_path)
    
    video = torch.nn.functional.interpolate(video, size=(244, 244), mode='bilinear', align_corners=False)
    audio_fft = torch.randn(1,2,3)
    video_fft = []
    for i in range(len(video)):
        frame = video[i].numpy().transpose(1, 2, 0)  # Convert to HxWxC
        gray_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray_frame)
        magnitude_image, ifft_image = fft_to_image_and_ifft(f)
        out_width = 244
        out_height = 244
        magnitude_image = cv2.resize(ifft_image, (out_width, out_height))
        color_mapped_image = cv2.applyColorMap(magnitude_image, cv2.COLORMAP_JET)
        video_fft.append(color_mapped_image)
    video_fft = torch.FloatTensor(video_fft)
 
    labels_list = ["detect whether this video is real or fake?", " Can you make sure it is fake? ", "I think it is fake."]
    
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name)#.to("cuda:3")
    clip_model = clip_model.to(device)
    
    inputs = processor(text=labels_list, images=video, return_tensors="pt", padding=True, do_rescale=False)
    inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}
    outputs = clip_model(**inputs)
    
    video_embeddings =  outputs.image_embeds.cpu().detach().numpy()
    text_embeddings =  outputs.text_embeds.cpu().detach().numpy()
    
    audio = audio.permute(1,0)
    audio = audio.mean(dim=0)  # Convert to mono
    audio = pad_random(audio, max_len=64000)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    video_embeddings = torch.from_numpy(video_embeddings)
    text_embeddings = torch.from_numpy(text_embeddings)
    audio, video, audio_fft, video_fft, text_embeddings, video_embeddings = audio.unsqueeze(0), video.unsqueeze(0),audio_fft.unsqueeze(0), video_fft.unsqueeze(0), text_embeddings.unsqueeze(0), video_embeddings.unsqueeze(0)
    audio = audio.repeat(2, 1)  # Repeat the batch size
    video = video.repeat(2, 1, 1, 1, 1)  # Repeat the batch size
    audio_fft = audio_fft.repeat(2, 1, 1, 1)  # Repeat the batch size
    video_fft = video_fft.repeat(2, 1, 1, 1, 1)  # Repeat the batch size
    text_embeddings = text_embeddings.repeat(2, 1, 1)  # Repeat the batch size
    video_embeddings = video_embeddings.repeat(2, 1, 1)  # Repeat the batch size
    video_fft = video_fft.permute(0,1,4,2,3)
    print(audio.shape, video.shape, audio_fft.shape, video_fft.shape, text_embeddings.shape, video_embeddings.shape)
    with torch.no_grad():
        logits = model(audio, video, audio_fft, video_fft, text_embeddings, video_embeddings)
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
    
    filenames = [video_path]
    
    first_probability = probabilities[0]

    # Determine the class prediction for the first sample
    pred_class = "Fake" if first_probability > 0.5 else "Real"

    # Calculate probabilities for being Fake or Real
    fake_prob = first_probability.item() * 100  # Convert to percentage
    real_prob = (1 - first_probability).item() * 100  # Convert to percentage

    # for f,r, filename in zip(fake_prob, real_prob, video_path):
    #         with open(os.path.join("scores", f'scorestmit.txt'), "a") as f:
    #             f.write(f"{filename}, {f}, {r}\n")
    print(f"First Sample Prediction: {pred_class}, Fake Probability: {fake_prob:.2f}%, Real Probability: {real_prob:.2f}%")
    # pred_classes = (probabilities > 0.5).float()  
    # pred_class_labels = ["Fake" if pred > 0.5 else "Real" for pred in probabilities]

    # # Calculate probabilities for each class
    # fake_probs = probabilities * 100  # Percentage of being fake
    # real_probs = (1 - probabilities) * 100  # Percentage of being real

    # # Display predictions
    # for i, (cls, fake_prob, real_prob) in enumerate(zip(pred_class_labels, fake_probs, real_probs)):

    #     print(f"Sample {i}: Prediction: {cls}, Fake Probability: {fake_prob.item()}%, Real Probability: {real_prob.item()}%")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42, help="The random seed.")
    parser.add_argument("--model_path", type=str,default = f"/home/shahid/LAV-DF-Shahid/logs/mfcc/20240728-180212-100%acc_DFDC_voxceleb/checkpoints/model_10_EER_0.0_loss_0.0017890970155994797_accuracy_1.0.pt", help="The path to the model.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--encoder", type=str, default="mfcc", help="The encoder to use.")
    
    args = parser.parse_args()
    # /media/NAS/DATASET/DeepfakeTIMIT/DeepfakeTIMIT/ directory = "/media/NAS/DATASET/DeepfakeTIMIT/DeepfakeTIMIT/lower_quality"

    # List to store all .avi file paths
    avi_files = []

    directory = "/media/NAS/DATASET/DFDC-Official/dfdc_preview_set"

    # Walk through the directory and its subdirectories
    # for root, _, files in os.walk(directory):
    #     for file in files:
    #         # Check if the file is an .avi file
    #         if file.endswith('.avi'):
    #             # Construct the full path to the .avi file
    #             avi_files.append(os.path.join(root, file))
    # for video_path in avi_files:  # Specify the path to your video file
    video_path = "/home/shahid/LAV-DF-Shahid/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/Asian (East)/men/id00056/00028_id02332_wavtolip.mp4"
    single_video_inference(args, video_path)
