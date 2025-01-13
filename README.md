# Whisper Speaker Identification (WSI)

**Whisper Speaker Identification (WSI)** is a state-of-the-art speaker identification model designed for multilingual scenarios.The WSI model adapts OpenAI's Whisper encoder and fine-tunes it with a projection head using triplet loss-based metric learning. This approach enhances its ability to generate discriminative, language-agnostic speaker embeddings.WSI demonstrates state-of-the-art performance on multilingual datasets, achieving lower Equal Error Rates (EER) and higher F1 Scores compared to models such as **pyannote/wespeaker-voxceleb-resnet34-LM** and **speechbrain/spkrec-ecapa-voxceleb**.

## Installation

Install the `whisper-speaker-id` library via pip:

```
pip install whisper-speaker-id 
```

## Usage

The `wsi` library provides a simple interface to use the WSI model for embedding generation and speaker similarity tasks. 

## Download the model from Huggingface

[WSI Model on Hugging Face](https://huggingface.co/emon-j/WSI)

### Generate Speaker Embeddings  

```python
from wsi import load_model, process_single_audio

# Load the model and feature extractor
model, feature_extractor = load_model("path/to/wsi_model.pth")

# Generate embeddings for an audio file
embedding = process_single_audio(model, feature_extractor, "path/to/audio.wav")
print("Speaker Embedding:", embedding)
```

### Calculate Similarity Between Two Audio Files

```python
from wsi import load_model, process_audio_pair

# Load the model and feature extractor
model, feature_extractor = load_model("path/to/wsi_model.pth")

# Compute similarity between two audio files
similarity = process_audio_pair(
    model, feature_extractor, "path/to/audio1.wav", "path/to/audio2.wav"
)
print("Similarity Score:", similarity)
```

### Cite This Work

Comming Soon!

### License

This project is licensed under the CC BY-NC-SA 4.0 License.
