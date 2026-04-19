import os
import io
import json
import base64
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import onnxruntime as ort

# Configuration
TOKEN = "55172888"
API_HOST = "http://34.122.51.94"
PORT = "9306"
SEED = "74714394"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= API Query =======================
def model_stealing(images, port):
    url = f"{API_HOST}:{port}/query"
    payload = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format='PNG'); buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        payload.append(b64)
    resp = requests.get(url, files={"file": json.dumps(payload)}, headers={"token": TOKEN})
    resp.raise_for_status()
    return resp.json().get("representations")

# ======================= Proxy Dataset =======================
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

class PtProxyDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        data = torch.load(pt_path, weights_only=False)
        assert isinstance(data, TaskDataset)
        self.raw_images = data.imgs
        self.transform = transform
    def __len__(self): return len(self.raw_images)
    def __getitem__(self, idx):
        raw = self.raw_images[idx]
        pil = transforms.ToPILImage()(raw) if isinstance(raw, torch.Tensor) else raw
        pil = pil.convert('RGB')
        tensor = self.transform(pil) if self.transform else transforms.ToTensor()(pil)
        return tensor, pil

# ======================= Student Model =======================
class StolenResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18(pretrained=False)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 1024)
    def forward(self, x):
        return self.net(x)

# ======================= Training Setup =======================
class StealDataset(Dataset):
    def __init__(self, orig_tensors, teacher_reps, pil_images):
        self.orig = orig_tensors
        self.teach = teacher_reps
        self.pils = pil_images
    def __len__(self): return len(self.orig)
    def __getitem__(self, idx):
        return self.orig[idx], self.teach[idx], self.pils[idx]

# ======================= Main Execution =======================
    
def collate_fn(batch):
    orig_batch = torch.stack([item[0] for item in batch])
    teacher_batch = torch.stack([item[1] for item in batch])
    pils = [item[2] for item in batch]  # list of PIL.Image
    return orig_batch, teacher_batch, pils

def main():
    pt_file = "ModelStealingPub.pt"
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    base_tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), norm])
    ds = PtProxyDataset(pt_file, transform=base_tf)

    idxs = torch.randperm(len(ds))[:1000]
    orig_tensors, pil_batch = [], []
    for i in idxs:
        t, p = ds[i]; orig_tensors.append(t); pil_batch.append(p)
    input_batch = torch.stack(orig_tensors)

    print("Querying teacher API for 1000 images...")
    teacher_reps = model_stealing(pil_batch, PORT)
    teacher_reps = torch.tensor(teacher_reps, dtype=torch.float32)

    aug_tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), norm
    ])

    dataset = StealDataset(orig_tensors, teacher_reps, pil_batch)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    model = StolenResNet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    lambda_aug = 0.5

    for epoch in range(50):
        model.train(); total_loss = 0.0
        for orig, teach, pils in loader:
            orig, teach = orig.to(device), teach.to(device)
            aug = torch.stack([aug_tf(pil) for pil in pils]).to(device)
            out1 = model(orig)
            out2 = model(aug)
            loss = criterion(out1, teach) + lambda_aug * criterion(out2, teach)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * orig.size(0)
        print(f"Epoch {epoch+1}/50: Loss={total_loss/len(dataset):.4f}")

    dummy = torch.randn(1, 3, 32, 32, device=device)
    torch.onnx.export(model, dummy, "stolen_encoder.onnx",
                      input_names=["x"], output_names=["output"], opset_version=11)
    sess = ort.InferenceSession("stolen_encoder.onnx")
    out = sess.run(None, {"x": np.random.randn(1, 3, 32, 32).astype(np.float32)})[0]
    assert out.shape == (1, 1024)
    print("ONNX model verified successfully.")

    with open("stolen_encoder.onnx", 'rb') as f:
        res = requests.post(f"{API_HOST}:9090/stealing", files={"file": f},
                            headers={"token": TOKEN, "seed": SEED})
    print("Submission response:", res.json())

if __name__ == "__main__":
    main()
