# SAMM-D: Enhancing DeepFake Detection with Subjective Assessment and Multimodal Fusion
This repository contains the system implementations for the SAMM-D submission to AAAI 2025. 

# About
DeepFake is becoming common in content creation such as avatar video translations and Virtual Human Assistants. Video DeepFake is generated using Deep Generative Artificial Intelligence (Generative AI) models by either swapping the face of the target with the source or replacing lip movements to match the source audio. Similarly, audio DeepFake clones the voices from source audio to target audio, making it difficult for humans to differentiate the difference. In previous works, DeepFake detection works by either detecting or manipulating Artifacts in images, videos, or audio using uni-model or multi-modal models. However, there are few works using text to guide the model through user prompts. Visual Language Model (VLM) give users the ability to interact guide and query the AI model through prompts. In this work, we propose a user-interactive Vision Machine 
Learning (VML) SAMM-D. By simultaneously learning representations of video in the frequency domain of video, audio, and text, our method grants users control and interactive capabilities.  We employed a pre-trained CLIP text encoder for making the user assessment and video representation that correlates with each modality to generate similarity with the corresponding user prompt to detect DeepFake.This is achieved by equipping users with the capability to guide the model through discrepancies and identify fake artifacts. The incorporation of prompt-guided multi-modal fusion resulted in notable enhancements in accuracy, Average Precision (AP), and the Area Under the Curve (AUC) for the detection of DeepFakes on the FakeAVCeleb training set and DFDC testing set. The proposed approach exhibited the highest accuracy and AUC, reaching 100%. Moreover, the efficiency of the approach was enhanced through a subjective assessment conducted during the inference stage.

# Getting Started

Setting up environment:
```bash
conda create -n svdd_baseline python=3.10
conda activate svdd_baseline
pip install -r requirements.txt
```

Then you can run the training script with the following command:
```bash
python train_fft.py --base_dir {Where the data is} --gpu {GPU ID} --encoder {Encoder Type} --batch_size {Batch size}
```
You can use `--load_from` flag to resume training.

After training, you can evaluate your model using the following command:
```bash
python eval.py --base_dir {Where the data is} --model_path {The model's weights file} --gpu {GPU ID} --encoder {Encoder Type} --batch_size {Batch size}
```

The main functions in `train` and `eval` specify more options that you can tune. 

Within `base_dir`, the code expects to see `train_set`, `dev_set` and `test_set` directories, along with `train.txt` and `dev.txt` as open-sourced. `train_set`, `dev_set` and `test_set` should directly contain `*.flac` files.

# Visualize Training Logs of Provided Baseline Systems
Run the following command within the CtrSAMMD_dataset_Baseline directory.

```bash
pip install tensorboard
tensorboard --logdir weights/training_logs
```

