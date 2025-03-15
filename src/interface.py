import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WhisperModel
from torch.nn import Module, Sequential, Linear, ReLU
from scipy.spatial.distance import cosine
from huggingface_hub import hf_hub_download
import os 
import numpy as np

AUDIO_SAMPLING_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderOnlyEmbeddingExtractor(Module):
    def __init__(self, whisper_model, embed_dim):
        super().__init__()
        self.encoder = whisper_model.encoder  
        self.projection = Sequential(
            Linear(whisper_model.config.d_model, embed_dim),
            ReLU(),
            Linear(embed_dim, embed_dim)
        )

    def forward(self, input_features):
        encoder_states = self.encoder(input_features).last_hidden_state  
        pooled = encoder_states.mean(dim=1)
        embeddings = self.projection(pooled)
        return embeddings

def load_model(model_path_or_repo_id, embed_dim=256, model_id="openai/whisper-tiny", filename="wsi.pth"):
    if not os.path.isfile(model_path_or_repo_id):
        model_path_or_repo_id = hf_hub_download(repo_id=model_path_or_repo_id, filename=filename)

    whisper_model = WhisperModel.from_pretrained(model_id).to(DEVICE)
    model = EncoderOnlyEmbeddingExtractor(whisper_model, embed_dim).to(DEVICE)
    state_dict = torch.load(model_path_or_repo_id, map_location=DEVICE)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    return model, feature_extractor

def load_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != AUDIO_SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLING_RATE)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy()

def get_embedding(model, feature_extractor, audio_path):
    audio_array = load_audio(audio_path)
    with torch.no_grad():
        input_features = feature_extractor(audio_array, sampling_rate=AUDIO_SAMPLING_RATE, return_tensors="pt").input_features.to(DEVICE)
        embedding = model(input_features).squeeze(0).cpu().numpy()
    return embedding

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding if norm == 0 else embedding / norm

def calculate_similarity(embedding1, embedding2):
    embedding1 = normalize_embedding(embedding1)
    embedding2 = normalize_embedding(embedding2)
    return 1 - cosine(embedding1, embedding2)

def process_single_audio(model, feature_extractor, audio_path):
    return get_embedding(model, feature_extractor, audio_path)

def process_audio_pair(model, feature_extractor, audio_path1, audio_path2):
    embedding1 = get_embedding(model, feature_extractor, audio_path1)
    embedding2 = get_embedding(model, feature_extractor, audio_path2)
    return calculate_similarity(embedding1, embedding2)
