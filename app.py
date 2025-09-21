import os
import gdown
import torch
import pickle

MODEL_PATH = "bilstm_char_model_state.pth"
SRC_VOCAB = "src2id.pkl"
TGT_VOCAB = "tgt2id.pkl"

# 🔹 Google Drive File ID of your model
MODEL_DRIVE_ID = "1F9AnNf51jhRl4wVozn3po7k0pSMi7ON5"   

# Download from Google Drive if model not already present
if not os.path.exists(MODEL_PATH):
    print("⏳ Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")

# Load model state
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
