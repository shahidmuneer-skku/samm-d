import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import datetime
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from datasets_lavdf import SAMMD_dataset
from models.model_clip_fft_lavdf import SAMMDModel
from util import seed_worker, set_seed, compute_eer

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc,precision_recall_curve
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits

    def forward(self, logits, targets):
        if self.use_logits:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def smooth_labels(labels, smoothing=0.1):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        labels = labels * (1 - smoothing) + smoothing / 2
    return labels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    # Set the seed for reproducibility
    set_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # resume-training flag, if set to True, the model, optimizer and scheduler will be loaded from the checkpoint
    resume_training = False
    if args.load_from is not None:
        resume_training = True

    # Create the dataset
    path = args.base_dir
    train_dataset = SAMMD_dataset(path, partition="train")
    val_dataset = SAMMD_dataset(path, partition="dev")
    # dev_dataset = SAMMD_dataset(path, partition="dev")
    # train_size = len(train_dataset)
    # split_size = int(train_size * 0.3)
    # remaining_size = train_size - split_size

    # Split the validation dataset
    # val_subset, train_subset = random_split(train_dataset, [split_size, remaining_size])

    # Combine the original training dataset with the 30% subset of the validation dataset
    # combined_train_dataset = ConcatDataset([train_dataset, val_subset])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=0, worker_init_fn=seed_worker)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size,  drop_last=True, shuffle=False, num_workers=0)

    # Create the model
    model = SAMMDModel(device, frontend=args.encoder)
    model = model.to(device)  # Move model to the primary device first
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4])  # Wrap the model with DataParallel
    total_parameters = count_parameters(model=model)
    print(f"Model created and it has parameters {total_parameters}")

    # Create the optimizer
    # lr = 3, wd = 1e-9
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    start_epoch = 0

    if resume_training:
        model_state = torch.load(os.path.join(args.load_from, "checkpoints", "model_state.pt"))
        model.load_state_dict(model_state['model_state_dict'])
        optimizer.load_state_dict(model_state['optimizer_state_dict'])
        scheduler.load_state_dict(model_state['scheduler_state_dict'])
        start_epoch = model_state['epoch']
        log_dir = args.load_from
    else:
        # Create the directory for the logs
        log_dir = os.path.join(args.log_dir, args.encoder)
        os.makedirs(log_dir, exist_ok=True)

    # get current time
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, current_time)
    os.makedirs(log_dir, exist_ok=True)

    # Create the summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # Create the directory for the checkpoints
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(str(vars(args)))

    criterion = BinaryFocalLoss()

    # criterion = 

    best_val_eer = 1.0
    l2_lambda = 0.001
    i=0
    init = False
    # Train the model
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pos_samples, neg_samples,all_preds, all_labels = [], [],[],[]
        correct_predictions = 0
        total_samples = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            if args.debug and i > 500:
                break
            audio, video, label, audio_fft, video_fft, text_embeddings, video_embeddings, filenames = batch
            audio = audio.to(device)
            video = video.to(device)
            audio_fft = audio_fft.to(device)
            video_fft = video_fft.to(device)
            text_embeddings = text_embeddings.to(device)
            video_embeddings = video_embeddings.to(device)

            # text = text.to(device)
            label = label.to(device)
            # text = torch.cuda.tensor("")
            # text = torch.tensor("Find artifcats in the video, and tell whether it is real or fake?").to(device)
            soft_label = smooth_labels(labels=label.float())
            # print(audio.shape)
            pred = model(audio, video, audio_fft, video_fft, text_embeddings, video_embeddings)
            # ensemble = sum(pred[:-1])/len(pred)
            # ensemble.detach_()


            loss = criterion(pred, soft_label.unsqueeze(1))
            
            pos_samples.append(pred[label == 1].detach().cpu().numpy())
            neg_samples.append(pred[label == 0].detach().cpu().numpy())
            pred_class = (pred > 0.5).long().squeeze()
            correct_predictions += (pred_class == label).sum().item()
            total_samples += label.size(0)
            # sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())

            
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
        scheduler.step()

        accuracy = correct_predictions / total_samples

        # Compute ROC AUC
        roc_auc = roc_auc_score(all_labels, all_preds)  # Assuming binary classification and class 1 probabilities

        # Compute Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        auc_pr = auc(recall, precision)
        ap_score = average_precision_score(all_labels, all_preds)

        writer.add_scalar("Train/AUC", roc_auc, epoch)
        writer.add_scalar("Train/AP", ap_score, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        writer.add_scalar("LR/train", scheduler.get_last_lr()[0], epoch * len(train_loader) + i)
        writer.add_scalar("EER/train", compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0], epoch)
        # save training state
        model_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }
        torch.save(model_state, os.path.join(checkpoint_dir, f"model_state.pt"))

        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_samples = 0
        pos_samples, neg_samples,all_preds, all_labels= [], [],[],[]
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dev_loader)):
                # if args.debug and i > 100:
                #     break
                audio, video, label, audio_fft, video_fft, text_embeddings, video_embeddings, filenames = batch
                audio = audio.to(device)
                video = video.to(device)
                label = label.to(device)
                audio_fft = audio_fft.to(device)
                video_fft = video_fft.to(device)
                text_embeddings = text_embeddings.to(device)
                video_embeddings = video_embeddings.to(device)

                # print(audio.shape,video.shape)
                # text = text.to(device)
                # mask = torch.ones(1, 1024).bool()
                # mask = mask.to(device)
                pred = model(audio, video, audio_fft, video_fft, text_embeddings, video_embeddings)

                # Assuming binary classification and pred is the output probability
                pred_class = (pred > 0.5).long().squeeze()
                correct_predictions += (pred_class == label).sum().item()
                total_samples += label.size(0)

                # soft_label = label.float() * 0.9 + 0.05
                soft_label = smooth_labels(labels=label.float())
                loss = criterion(pred, soft_label.float().unsqueeze(1))
                pos_samples.append(pred[label == 1].detach().cpu().numpy())
                neg_samples.append(pred[label == 0].detach().cpu().numpy())
                all_preds.extend(pred.detach().cpu().numpy())
                all_labels.extend(label.detach().cpu().numpy())
                val_loss += loss.item()
            val_loss /= len(dev_loader)
            accuracy = correct_predictions / total_samples

            val_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
            # roc_auc = roc_auc_score(all_labels, all_preds)
            # Compute ROC AUC
            roc_auc = roc_auc_score(all_labels, all_preds)  # Assuming binary classification and class 1 probabilities

            # Compute Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            auc_pr = auc(recall, precision)
            ap_score = average_precision_score(all_labels, all_preds)

            writer.add_scalar("Val/AUC", roc_auc, epoch)
            writer.add_scalar("Val/AP", ap_score, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("EER/val", val_eer, epoch)
            writer.add_scalar("Accuracy/val", accuracy, epoch)

            if val_eer < best_val_eer:
                best_val_eer = val_eer
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model.pt"))

            if epoch % 5 == 0:  # Save every 20 epochs
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}_EER_{val_eer}_loss_{val_loss}_accuracy_{accuracy}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs to train.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--encoder", type=str, default="rawnet", help="The encoder to use.")
    parser.add_argument("--batch_size", type=int, default=24, help="The batch size for training.")
    parser.add_argument("--num_workers", type=int, default=6, help="The number of workers for the data loader.")
    parser.add_argument("--log_dir", type=str, default="logs", help="The directory for the logs.")
    parser.add_argument("--load_from", type=str, default=None, help="The path to the checkpoint to load from.")

    args = parser.parse_args()
    main(args)
