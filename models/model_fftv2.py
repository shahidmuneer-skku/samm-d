"""
This code is largely modified from the codebase of AASIST.
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import random
from typing import Union
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from models.hubertClassifier import HubertLSTMClassifier
from models.backend import AASIST
# try:
import torchvision.models as models
# from s3prl.nn import S3PRLUpstream
from transformers import CLIPProcessor, CLIPModel
from open_clip import create_model_from_pretrained, get_tokenizer 

# except NotFoundError:
#     S3PRLUpstream = None
# import fairseq
import argparse
from sklearn.decomposition import PCA
from models.transformer_encoder import TransformerEncoder
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        # print('out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        # print('conv2 out',out.shape)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
    ):
        super().__init__()
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        if in_channels != 1:

            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(
            -(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1
        )
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate
            )
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp / self.sample_rate
            )
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

    def forward(self, x):
        band_pass_filter = self.band_pass.clone().to(x.device)
        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        x = x.unsqueeze(1)
        x = F.conv1d(
            x,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        # get embeddings using encoder
        # (#bs, #filt, #spec, #seq)
        x = self.encoder(x)
        return x

class Spectrogram(nn.Module):
    def __init__(self, device, sample_rate=16000, n_fft=512, win_length=512, hop_length=160, power=2, normalized=True):
        super(Spectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.power = power
        self.normalized = normalized
        
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power,
            normalized=self.normalized,
        ).to(device)
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.linear = nn.Linear((n_fft // 2 + 1) * 4, 23 * 29) # match the output shape of the rawnet encoder

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.spec(x)
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        return x
    
class MelSpectrogram(nn.Module):
    def __init__(self, device, sample_rate=16000, n_mels=80, n_fft=512, win_length=512, hop_length=160):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
        ).to(device)
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.linear = nn.Linear(n_mels * 4, 23 * 29) # match the output shape of the rawnet encoder

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.melspec(x)
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        return x
    
class LFCC(nn.Module):
    def __init__(self, device, sample_rate=16000, n_filter=20, f_min=0.0, f_max=None, n_lfcc=60, dct_type=2, norm="ortho", log_lf=False, speckwargs={"n_fft": 512, "win_length": 512, "hop_length": 160, "center": False}):
        super(LFCC, self).__init__()
        self.sample_rate = sample_rate
        self.n_filter = n_filter
        self.f_min = f_min
        self.f_max = f_max
        self.n_lfcc = n_lfcc
        self.dct_type = dct_type
        self.norm = norm
        self.log_lf = log_lf
        self.speckwargs = speckwargs
        
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=self.sample_rate,
            n_filter=self.n_filter,
            f_min=self.f_min,
            f_max=self.f_max,
            n_lfcc=self.n_lfcc,
            dct_type=self.dct_type,
            norm=self.norm,
            log_lf=self.log_lf,
            speckwargs=self.speckwargs,
        ).to(device)
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.linear = nn.Linear(n_lfcc * 4, 23 * 29) # match the output shape of the rawnet encoder

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.lfcc(x)
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        return x
    
class MFCC(nn.Module):
    def __init__(self, device, sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 512, "win_length": 512, "hop_length": 160, "center": False}):
        super(MFCC, self).__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.melkwargs = melkwargs
        
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs=self.melkwargs,
        ).to(device)
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.linear = nn.Linear(n_mfcc * 4, 23 * 29) # match the output shape of the rawnet encoder

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mfcc(x)
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        # x = x.view(x.size(0), x.size(1), 23, 29)
        return x
    
class SSLFrontend(nn.Module):
    def __init__(self, device, model_label, model_dim):
        super(SSLFrontend, self).__init__()
        if model_label == "xlsr":
            task_arg = argparse.Namespace(task='audio_pretraining')
            task = fairseq.tasks.setup_task(task_arg)
            # https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt
            model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(['/root/xlsr2_300m.pt'], task=task)
            self.model = model[0]
        self.device = device
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]

        self.sample_rate = 16000 # only 16000 setting is supported
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        self.linear = nn.Linear(model_dim * 2, 23 * 29)
        
    def extract_feature(self, x):
        if next(self.model.parameters()).device != x.device \
            or next(self.model.parameters()).dtype != x.dtype:
            self.model.to(x.device, dtype=x.dtype)
            self.model.train()
        emb = self.model(x, mask=False, features_only=True)['x']
        return emb
    
    def forward(self, x):
        x = self.extract_feature(x)
        x = x.transpose(1, 2).unsqueeze(1) # [batch, 1, seq, dim]
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        return x


class S3PRL(nn.Module):
    def __init__(self, device, model_label, model_dim):
        super(S3PRL, self).__init__()
        if S3PRLUpstream is None:
            raise ModuleNotFoundError("s3prl is not found, likely not installed, please install use `pip`")

        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]

        self.sample_rate = 16000 # only 16000 setting is supported
        if model_label == "mms":
            self.model = S3PRLUpstream(
                "hf_wav2vec2_custom",
                path_or_url="facebook/mms-300m",
            ).to(device)
            print("Model has been sent to", device)
        else:
            self.model = S3PRLUpstream(model_label).to(device)
            print("Model has been sent to", device)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        self.linear = nn.Linear(model_dim * 2 * 64, 1) # match the output shape of the rawnet encoder

    def forward(self, x):
        # print(x.size()) # expected: torch.Size([batch, 64000])
        # x_lens = torch.LongTensor(x.size(0)).to(x.device)
        # x, _ = self.model(x, x_lens)
        # x = x.
        x = x[-1].transpose(1, 2).unsqueeze(1) # take the last hidden states
        print(x.size())
        x = self.encoder(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear(x)
        x = x.view(x.size(0), 1)
        return x

class SAMMDModel(nn.Module):
    def __init__(self, device, frontend=None):
        super(SAMMDModel, self).__init__()
        assert frontend in ["rawnet", "spectrogram", "mel-spectrogram", "lfcc", "mfcc", "hubert", "mms", "xlsr", "mrhubert", "wavlablm"], "Invalid frontend"
        if frontend == "rawnet":
            # This follows AASIST's implementation
            self.frontend = SincConv(out_channels=70, kernel_size=128, in_channels=1)
        elif frontend == "spectrogram":
            self.frontend = Spectrogram(
                device=device,
                sample_rate=16000,
                n_fft=512,
                win_length=512,
                hop_length=160,
                power=2,
                normalized=True,
            )
        elif frontend == "mel-spectrogram":
            self.frontend = MelSpectrogram(
                device=device,
                sample_rate=16000,
                n_mels=80,
                n_fft=512,
                win_length=512,
                hop_length=160,
            )
        elif frontend == "lfcc":
            self.frontend = LFCC(
                device=device,
                sample_rate=16000,
                n_filter=20,
                f_min=0.0,
                f_max=None,
                n_lfcc=60,
                dct_type=2,
                norm="ortho",
                log_lf=False,
                speckwargs={
                    "n_fft": 512,
                    "win_length": 512,
                    "hop_length": 160,
                    "center": False,
                },
            )
        elif frontend == "mfcc":
            self.frontend = MFCC(
                device=device,
                sample_rate=16000,
                n_mfcc=40,
                melkwargs={
                    "n_fft": 512,
                    "win_length": 512,
                    "hop_length": 160,
                    "center": False,
                },
            )
        elif frontend == "hubert":
            self.frontend = S3PRL(
                device=device,
                model_label="hubert",
                model_dim=768,
            )
        elif frontend == "xlsr":
            self.frontend = SSLFrontend(
                device=device,
                model_label="xlsr",
                model_dim=1024,
            )
            print("after frontend")
        elif frontend == "mrhubert":
            self.frontend = S3PRL(
                device=device,
                model_label="multires_hubert_multilingual_large600k",
                model_dim=1024,
            )

        self.backend = AASIST(device).to(device)
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")#.to(device)
        
        # model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14')
        # self.tokenizer = get_tokenizer('ViT-H-14')
        # self.model = model 
        # self.preprocess = preprocess

        # for param in self.model.parameters():
        #     param.requires_grad = False
        # self.audio_encoder = ESResNeXtFBSP(n_fft=512,
        #                     hop_length=392,
        #                     win_length=430,
        #                     window="blackmanharris",
        #                     normalized=True,
        #                     onesided=True,
        #                     spec_height=-1,
        #                     spec_width=-1,
        #                     num_classes=1,
        #                     apply_attention=True,
        #                     pretrained="/home/shahid/DeepfakeDetection/PromptCLIP/ESResNeXtFBSP_AudioSet.pt")
        # self.reducer = nn.Linear(512,128)
        # CNN for audio embeddings
        # self.downsample_video =  nn.Sequential(
        #     # First 3D Convolutional layer
        #             nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
        #             nn.ReLU(),
        #             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

        #             # Second 3D Convolutional layer
        #             nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=1),
        #             nn.ReLU(),
        #             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

        #             # Third 3D Convolutional layer
        #             nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=2),
        #             nn.ReLU(),
        #             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 3)),
        #         )
        self.downsample_video = nn.Sequential(
            # First 3D Convolutional layer
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            # Second 3D Convolutional layer
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            # Third 3D Convolutional layer
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2)),
        )
        self.cnn = nn.Sequential(
            # First layer
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Second layer
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Third layer
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Fourth layer
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.audio_cnn = models.resnet18(pretrained=True)
        
        self.conv1d = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv1d_bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        # Adjust the first convolutional layer
        self.audio_cnn.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.audio_cnn.fc = nn.Linear(self.audio_cnn.fc.in_features, 512)  # Adjust the output size if needed
        self.audio_cnn = self.audio_cnn.to(device) # Define an LSTM layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True)
        # self.dropout = nn.Dropout(0.5)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=128),
            num_layers=6
        )
        self.transformer_fusion_video = TransformerEncoder(embed_dim=512,
                                                 num_heads=8,
                                                 layers=4,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.2,
                                                 res_dropout=0.2,
                                                 embed_dropout=0.25,
                                                 attn_mask=True)

        self.transformer_fusion_audio = TransformerEncoder(embed_dim=512,
                                                 num_heads=8,
                                                 layers=4,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.2,
                                                 res_dropout=0.2,
                                                 embed_dropout=0.25,
                                                 attn_mask=True)
        

        

        self.transformer_fusion = TransformerEncoder(embed_dim=512,
                                                 num_heads=8,
                                                 layers=4,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.2,
                                                 res_dropout=0.2,
                                                 embed_dropout=0.25,
                                                 attn_mask=True)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.linear = nn.Linear(512,256)
        # self.gelu = nn.GELU()
        # self.linear2 = nn.Linear(256,256)
        # self.gelu2 = nn.GELU()
        # self.linear3 = nn.Linear(256,128)
        # self.gelu3 = nn.GELU()
        # self.linear4 = nn.Linear(128,1)
        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Batch Normalization before activation
            nn.ELU(),  # Leaky ReLU
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),  # Batch Normalization before activation
            nn.ELU(),  # Leaky ReLU
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch Normalization before activation
            nn.ELU(),  # Leaky ReLU
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # self.backend = HubertLSTMClassifier('facebook/hubert-large-ls960-ft', lstm_hidden_size=128)
    




    def forward(self,audio, video, audio_fft, video_fft, text_embeddings, video_embeddings):
        # x = self.frontend(audio)
        audio_features = self.frontend(audio)
        audio_embeddings = self.cnn(audio_features)
        audio_embeddings = audio_embeddings.permute(0,2,1)
        audio_embeddings = self.transformer_encoder(audio_embeddings)
        

        video_fft = video_fft.permute(0,2,1,3,4)
        video_fft = self.downsample_video(video_fft)
        video_fft = torch.flatten(video_fft,start_dim=2, end_dim=4)
        video_fft = F.interpolate(video_fft, size=(512), mode='linear').squeeze(-1)
        video_embeddings = video_embeddings.squeeze().permute(1,0,2).contiguous()
        video_fft = video_fft.squeeze().permute(1,0,2).contiguous()
        text_embeddings = text_embeddings.permute(1,0,2).contiguous()
        audio_embeddings = audio_embeddings.squeeze().permute(1,0,2).contiguous()


        # video_to_text = self.transformer_fusion_video(video_embeddings, text_embeddings,text_embeddings)
        
        # video_to_text = self.transformer_fusion_video(video_embeddings, text_embeddings,text_embeddings)
        # text_to_video = self.transformer_fusion_video(text_embeddings, video_embeddings,video_embeddings)


        # audio_to_text = self.transformer_fusion_audio(audio_embeddings, text_embeddings,text_embeddings)
        # text_to_audio = self.transformer_fusion_audio(text_embeddings, audio_embeddings,audio_embeddings)

        # transformer_out_video = self.transformer_fusion_video(video_to_text, text_to_video,text_to_video)


        # transformer_out_audio = self.transformer_fusion_audio(audio_to_text, text_to_audio,text_to_audio)

        transformer_out_video  = self.transformer_fusion_video(video_embeddings, video_fft,video_fft)
        out = self.transformer_fusion(transformer_out_video,audio_embeddings,audio_embeddings)
        t = out.shape[0]
        out = F.max_pool1d(out.permute(1, 2, 0).contiguous(), t).squeeze(-1)
        x = self.linear(out)
        x = torch.sigmoid(x)  
        # x = self.gelu(self.linear(lstm_out))
        # x = self.gelu2(self.linear2(x))
        # x = self.gelu3(self.linear3(x))
        # x = self.linear4(x)

        return x

if __name__ == "__main__":
    x = torch.randn(4, 64000)
    
    print("Testing RawNet Encoder")
    model = SAMMDModel(frontend="rawnet")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 2])
    
    print("Testing Spectrogram Encoder")
    model = SAMMDModel(frontend="spectrogram")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 1])
    
    print("Testing Mel-Spectrogram Encoder")
    model = SAMMDModel(frontend="mel-spectrogram")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 1])
    
    print("Testing LFCC Encoder")
    model = SAMMDModel(frontend="lfcc")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 1])
    
    print("Testing MFCC Encoder")
    model = SAMMDModel(frontend="mfcc")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 1])