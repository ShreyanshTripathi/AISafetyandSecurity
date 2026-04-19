import os
import sys
import io
import json
import base64
import pickle
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import onnxruntime as ort
import torch.onnx
from PIL import Image

# ====================================================================
# Configuration and API setup
# ====================================================================
TOKEN = "55172888"  # Replace with your token
API_HOST = "http://34.122.51.94"
# Reuse existing API port & seed to avoid launch limits
PORT = "9306"    # your port
SEED = "74714394"  # your seed
print(f"Using API on port {PORT} with seed {SEED}")

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ====================================================================
# TaskDataset stub (for unpickling)
# ====================================================================
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

# ====================================================================
# API Query Function
# ====================================================================
def model_stealing(images, port):
    """
    Query the remote encoder API with a list of PIL Images.
    Returns a list of 1024-d representations.
    """
    url = f"{API_HOST}:{port}/query"
    payload = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        payload.append(b64)
    resp = requests.get(url, files={"file": json.dumps(payload)}, headers={"token": TOKEN})
    if resp.status_code != 200:
        raise RuntimeError(f"Query failed: {resp.status_code}, {resp.text}")
    return resp.json().get("representations")

# ====================================================================
# Proxy Dataset Loader (.pt file using TaskDataset)
# ====================================================================
class PtProxyDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        # weights_only=False allows loading TaskDataset
        data = torch.load(pt_path, weights_only=False)
        assert isinstance(data, TaskDataset)
        self.raw_images = data.imgs  # list of raw objects (Tensor or PIL.Image)
        self.transform = transform

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw = self.raw_images[idx]
        # ensure PIL.Image
        if isinstance(raw, torch.Tensor):
            pil = transforms.ToPILImage()(raw)
        else:
            pil = raw
        # convert any mode to RGB
        pil = pil.convert('RGB')
        # apply transform to get tensor
        tensor = self.transform(pil) if self.transform else transforms.ToTensor()(pil)
        return tensor, pil

# ====================================================================
# Student Model Definition
# ====================================================================
class StolenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, 1024)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ====================================================================
# Main Execution: Single Query + Local Training
# ====================================================================
def main():
    # 1. Load full proxy dataset
    pt_file = "ModelStealingPub.pt"
    # Use minimal transforms for student inputs
    transforms_norm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    ds = PtProxyDataset(pt_file, transform=transforms_norm)

    # 2. Sample 1000 random indices
    N = len(ds)
    if N < 1000:
        raise RuntimeError(f"Dataset too small: {N} images")
    idxs = np.random.choice(N, 1000, replace=False)
    input_tensors = []
    pil_batch = []
    for i in idxs:
        t, p = ds[i]
        input_tensors.append(t)
        pil_batch.append(p)
    input_batch = torch.stack(input_tensors, dim=0).to(device)

    # 3. Single API query for teacher embeddings
    print("Querying teacher API for 1000 images...")
    teacher_reps = model_stealing(pil_batch, PORT)
    teacher_reps = torch.tensor(teacher_reps, dtype=torch.float32, device=device)

    # 4. Initialize student model
    student = StolenEncoder().to(device)

    # 5. Train student locally on this batch
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 10
    for epoch in range(epochs):
        preds = student(input_batch)
        loss = criterion(preds, teacher_reps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 6. Export to ONNX
    onnx_path = "stolen_encoder.onnx"
    dummy = torch.randn(1, 3, 32, 32, device=device)
    torch.onnx.export(student, dummy, onnx_path,
                      input_names=["x"], output_names=["output"], opset_version=11)
    print(f"Saved ONNX model to {onnx_path}")

    # 7. Verify ONNX
    sess = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    out = sess.run(None, {"x": test_input})[0]
    assert out.shape == (1, 1024)
    print("ONNX verification passed")

    # 8. Submit stolen model
    with open(onnx_path, 'rb') as f:
        files = {"file": f}
        headers = {"token": TOKEN, "seed": SEED}
        resp = requests.post(f"{API_HOST}:9090/stealing", files=files, headers=headers)
    print("Submission response:", resp.json())

if __name__ == "__main__":
    main()
