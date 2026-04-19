import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import requests
from PIL import Image
from torch.serialization import add_safe_globals

# ---- Step 1: Define the missing TaskDataset class ----
class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, imgs=None, labels=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = None

    def __getitem__(self, idx):
        img = self.imgs[idx].convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.labels)

# Allow safe loading of this class
add_safe_globals({"TaskDataset": TaskDataset})


# ---- Step 2: Main training + submission pipeline ----
def main():
    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cpu")
    os.makedirs("out/models", exist_ok=True)

    # Training transform (augment)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor()
    ])

    # Test/Val transform (no aug)
    test_transform = transforms.ToTensor()

    # Load the dataset (safely!)
    dataset = torch.load("Train.pt", weights_only=False)

    # Split: 70/15/15
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # Assign transforms
    train_data.dataset.transform = train_transform
    val_data.dataset.transform = test_transform
    test_data.dataset.transform = test_transform

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # ---- Model: ResNet18 ----
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    # ---- Training loop ----
    best_acc = 0.0
    for epoch in range(7):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/20 - Val Accuracy: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "out/models/resnet18_best.pt")

    # ---- Final Evaluation on Test Set ----
    model.load_state_dict(torch.load("out/models/resnet18_best.pt", map_location=device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    test_acc = correct / total
    print(f"\nFinal Clean Test Accuracy: {test_acc*100:.2f}%")

    # ---- Save final model for submission ----
    final_path = "out/models/dummy_submission.pt"
    torch.save(model.state_dict(), final_path)

    # Format check before submission
    model_eval = resnet18(weights=None)
    model_eval.fc = nn.Linear(model_eval.fc.in_features, 10)
    model_eval.load_state_dict(torch.load(final_path, map_location=device))
    model_eval.eval()
    out = model_eval(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 10)

    # ---- Submit to server ----
    try:
        response = requests.post(
            "http://34.122.51.94:9090/robustness",
            files={"file": open(final_path, "rb")},
            headers={"token": "55172888", "model-name": "resnet18"}
        )
        print("Server response:", response.json())
    except Exception as e:
        print("Submission failed:", e)

if __name__ == "__main__":
    main()
