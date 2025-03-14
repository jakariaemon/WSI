import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, Audio
from transformers import AutoFeatureExtractor, WhisperModel
import random
import torch.nn.functional as F
import numpy as np
import wandb
from sklearn.metrics import roc_curve, auc
import soundfile as sf
from io import BytesIO

# Configs
MODEL_ID = "openai/whisper-tiny"
AUDIO_SAMPLING_RATE = 16000
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-5
EMBED_DIM = 256
MARGIN = 1.0
SELF_SUP_WEIGHT = 1.0  # Weight for self-supervised loss
TEMPERATURE = 0.5      # Temperature for NT-Xent loss
MODEL_SAVE_PATH = "speaker_embedding_model_online_triplet_multiview.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints_new"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize Logging
wandb.init(project="online-triplet-multiview", config={
    "model_id": MODEL_ID,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "embed_dim": EMBED_DIM,
    "margin": MARGIN,
    "self_sup_weight": SELF_SUP_WEIGHT,
    "temperature": TEMPERATURE
})

checkpoint_artifact = wandb.Artifact('speaker_checkpoints', type='model')

# Load Dataset (Fixed Train / Val / Test)
train_dataset_raw = load_from_disk("fixed_train")
val_dataset_raw = load_from_disk("fixed_val")
#test_dataset_raw = load_from_disk("fixed_test")
train_dataset_raw = train_dataset_raw.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE, decode=False))
val_dataset_raw = val_dataset_raw.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE, decode=False))

# Audio Validation 
def is_valid_audio(example):
    try:
        audio = example["audio"]
        if "path" in audio:
            sf.info(audio["path"])
        elif "bytes" in audio:
            sf.info(BytesIO(audio["bytes"]))
        else:
            return False  
        return True
    except Exception:
        return False

print("Filtering train dataset...")
valid_indices_train = [i for i, sample in enumerate(train_dataset_raw) if is_valid_audio(sample)]
print(f"Valid train samples: {len(valid_indices_train)} out of {len(train_dataset_raw)}")
train_dataset_raw = train_dataset_raw.select(valid_indices_train)

print("Filtering validation dataset...")
valid_indices_val = [i for i, sample in enumerate(val_dataset_raw) if is_valid_audio(sample)]
print(f"Valid validation samples: {len(valid_indices_val)} out of {len(val_dataset_raw)}")
val_dataset_raw = val_dataset_raw.select(valid_indices_val)

train_dataset_raw = train_dataset_raw.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE, decode=True))
val_dataset_raw = val_dataset_raw.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE, decode=True))

# mapping for speaker IDs
all_spk_ids = list(set(train_dataset_raw["spk_id"] + val_dataset_raw["spk_id"]))
speaker_to_id = {spk: idx for idx, spk in enumerate(all_spk_ids)}

def map_speaker(example):
    example["spk_id"] = speaker_to_id[example["spk_id"]]
    return example

train_dataset_raw = train_dataset_raw.map(map_speaker)
val_dataset_raw = val_dataset_raw.map(map_speaker)

NUM_SPEAKERS = len(speaker_to_id)
print(f"Number of unique speakers: {NUM_SPEAKERS}")

# Audio Augmentation Functions for Multi-View Self-Supervision
def augment_audio_noise(audio, noise_factor=0.005):
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    noise = torch.randn_like(audio_tensor) * noise_factor
    augmented = audio_tensor + noise
    return augmented.numpy()

def augment_audio_time(audio, rate=1.1):
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    orig_length = audio_tensor.shape[-1]
    new_length = max(1, int(orig_length / rate))
    audio_stretched = torch.nn.functional.interpolate(audio_tensor, size=new_length, mode='linear', align_corners=False)
    audio_stretched = audio_stretched.squeeze().numpy()
    if len(audio_stretched) < orig_length:
        padding = orig_length - len(audio_stretched)
        audio_stretched = np.pad(audio_stretched, (0, padding), mode='constant')
    else:
        audio_stretched = audio_stretched[:orig_length]
    return audio_stretched

# Online Triplet Mining with Multi-View Augmentation
class SpeakerDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            audio = self.dataset[idx]["audio"]["array"]
        except Exception as e:
            return None

        label = self.dataset[idx]["spk_id"]
        # Generate two augmented views: noise and time-stretched
        audio_noise = augment_audio_noise(audio)
        audio_time = augment_audio_time(audio)
        return audio, label, audio_noise, audio_time

train_speaker_dataset = SpeakerDataset(train_dataset_raw)
val_speaker_dataset = SpeakerDataset(val_dataset_raw)

# Collate Function
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

def pad_or_truncate(features, target_length=3000):
    if features.shape[-1] < target_length:
        padding = target_length - features.shape[-1]
        features = F.pad(features, (0, padding), mode="constant", value=0)
    elif features.shape[-1] > target_length:
        features = features[..., :target_length]
    return features

def speaker_collate_fn(batch):
    # Batch contains tuples: (audio, label, audio_noise, audio_time)
    audio_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]
    audio_noise_list = [item[2] for item in batch]
    audio_time_list = [item[3] for item in batch]

    features = feature_extractor(audio_list, sampling_rate=AUDIO_SAMPLING_RATE, return_tensors="pt", padding=True).input_features
    features_noise = feature_extractor(audio_noise_list, sampling_rate=AUDIO_SAMPLING_RATE, return_tensors="pt", padding=True).input_features
    features_time = feature_extractor(audio_time_list, sampling_rate=AUDIO_SAMPLING_RATE, return_tensors="pt", padding=True).input_features

    features = torch.stack([pad_or_truncate(f, target_length=3000) for f in features])
    features_noise = torch.stack([pad_or_truncate(f, target_length=3000) for f in features_noise])
    features_time = torch.stack([pad_or_truncate(f, target_length=3000) for f in features_time])
    labels = torch.tensor(label_list)
    return features, labels, features_noise, features_time

train_dataloader = DataLoader(train_speaker_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=speaker_collate_fn)
val_dataloader = DataLoader(val_speaker_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=speaker_collate_fn)

# Model Definition
class EmbeddingExtractor(nn.Module):
    def __init__(self, whisper_model, embed_dim):
        super().__init__()
        self.whisper = whisper_model
        d_model = self.whisper.config.d_model
        self.projection = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, input_features):
        batch_size = input_features.size(0)
        decoder_input_ids = torch.tensor([self.whisper.config.decoder_start_token_id] * batch_size).unsqueeze(1).to(input_features.device)
        outputs = self.whisper(input_features, decoder_input_ids=decoder_input_ids)
        encoder_states = outputs.encoder_last_hidden_state
        pooled = encoder_states.mean(dim=1)
        embeddings = self.projection(pooled)
        return embeddings

# Loss Functions
def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(batch_size).to(z1.device)
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.T, labels)
    return (loss_1 + loss_2) / 2

def batch_hard_triplet_loss(embeddings, labels, margin=MARGIN):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    loss = 0.0
    batch_size = embeddings.size(0)
    for i in range(batch_size):
        label = labels[i]
        mask_positive = (labels == label)
        mask_negative = (labels != label)
        # Exclude self from positive
        mask_positive[i] = False
        if mask_positive.sum() > 0:
            hardest_positive = pairwise_dist[i][mask_positive].max()
        else:
            hardest_positive = 0.0
        if mask_negative.sum() > 0:
            hardest_negative = pairwise_dist[i][mask_negative].min()
        else:
            hardest_negative = 0.0
        loss += torch.clamp(hardest_positive - hardest_negative + margin, min=0.0)
    loss = loss / batch_size
    return loss

# Load Whisper and Freeze Decoder
whisper_model = WhisperModel.from_pretrained(MODEL_ID).to(DEVICE)
whisper_model.config.output_hidden_states = True
for param in whisper_model.decoder.parameters():
    param.requires_grad = False

model = EmbeddingExtractor(whisper_model, EMBED_DIM).to(DEVICE)
optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)

# Evaluation Functions
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for features, labels, features_noise, features_time in dataloader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            embeddings = model(features)
            loss = batch_hard_triplet_loss(embeddings, labels, margin=MARGIN)
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
    return total_loss / total_samples

def calculate_metrics(model, dataloader):
    model.eval()
    scores = []
    true_labels = []
    with torch.no_grad():
        for features, labels, features_noise, features_time in dataloader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            embeddings = model(features)
            norm_emb = F.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.matmul(norm_emb, norm_emb.T)
            for i in range(embeddings.size(0)):
                for j in range(i+1, embeddings.size(0)):
                    scores.append(sim_matrix[i, j].item())
                    true_labels.append(1 if labels[i] == labels[j] else 0)
    scores = np.array(scores)
    true_labels = np.array(true_labels)
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    fnr = 1 - tpr
    diff = np.abs(fpr - fnr)
    eer = fpr[np.argmin(diff)]
    auc_score = auc(fpr, tpr)
    thresh = thresholds[np.argmin(diff)]
    preds = (scores >= thresh).astype(int)
    accuracy = (preds == true_labels).mean()
    return eer, auc_score, accuracy

# Training Loop with Combined Loss (Online Triplet + Multi-View Self-Supervised)
best_val_loss = float('inf')
best_epoch = -1

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_samples = 0
    for i, (features, labels, features_noise, features_time) in enumerate(train_dataloader):
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        features_noise = features_noise.to(DEVICE)
        features_time = features_time.to(DEVICE)
        optimizer.zero_grad()
        # Compute embeddings for original and both augmented views
        embeddings = model(features)
        embeddings_noise = model(features_noise)
        embeddings_time = model(features_time)
        # Online hard triplet loss
        triplet_loss = batch_hard_triplet_loss(embeddings, labels, margin=MARGIN)
        # Self-supervised NT-Xent losses for each augmented view
        self_sup_loss_noise = nt_xent_loss(embeddings, embeddings_noise, temperature=TEMPERATURE)
        self_sup_loss_time = nt_xent_loss(embeddings, embeddings_time, temperature=TEMPERATURE)
        self_sup_loss = (self_sup_loss_noise + self_sup_loss_time) / 2.0
        loss = triplet_loss + SELF_SUP_WEIGHT * self_sup_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
        if (i + 1) % 50 == 0:
            wandb.log({
                "train_step_loss": loss.item(),
                "triplet_loss": triplet_loss.item(),
                "self_sup_loss": self_sup_loss.item()
            })
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Step [{i+1}/{len(train_dataloader)}] | Loss: {loss.item():.4f}")
    train_avg_loss = total_loss / total_samples
    val_avg_loss = evaluate(model, val_dataloader)
    val_eer, val_auc, val_acc = calculate_metrics(model, val_dataloader)
    
    wandb.log({
        "train_loss": train_avg_loss,
        "val_loss": val_avg_loss,
        "val_eer": val_eer,
        "val_auc": val_auc,
        "val_acc": val_acc,
        "epoch": epoch+1
    })
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_avg_loss:.7f} | Val Loss: {val_avg_loss:.7f} | "
          f"Val EER: {val_eer:.7f} | Val AUC: {val_auc:.7f} | Val Acc: {val_acc:.7f}")
    
    # Save checkpoint after each epoch
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_avg_loss,
        'val_loss': val_avg_loss,
        'val_eer': val_eer,
        'val_auc': val_auc,
        'val_acc': val_acc
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    checkpoint_artifact.add_file(checkpoint_path)
    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        best_epoch = epoch + 1
        best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_avg_loss,
            'val_loss': val_avg_loss,
            'val_eer': val_eer,
            'val_auc': val_auc,
            'val_acc': val_acc
        }, best_checkpoint_path)
        print(f"New best model saved at epoch {best_epoch} with val_loss: {best_val_loss:.7f}")
        checkpoint_artifact.add_file(best_checkpoint_path)

wandb.log_artifact(checkpoint_artifact)
print("Training complete.")

# Save Final Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved at {MODEL_SAVE_PATH}")
