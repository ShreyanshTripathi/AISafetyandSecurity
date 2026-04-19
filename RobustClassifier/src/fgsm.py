# Robust CIFAR-10 Training Script using Adversarial Training
# (ResNet-18, FGSM/PGD attacks)
# Based on PyTorch tutorials and adversarial training literature.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import requests

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# CPU only; ensure deterministic behavior
torch.use_deterministic_algorithms(True)

# Device (CPU)
device = torch.device('cpu')

# Create output directory for model
os.makedirs("out/models", exist_ok=True)

# CIFAR-10 data transforms (to [0,1] range)
transform = transforms.Compose([transforms.ToTensor()])
# Load CIFAR-10 dataset
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

# Choose allowed model (ResNet-18 for CIFAR-10):contentReference[oaicite:0]{index=0}
model_name = "resnet18"
allowed_models = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
}
model = allowed_models[model_name](weights=None)  # ResNet-18 architecture:contentReference[oaicite:1]{index=1}
# Replace final fully-connected layer for 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Training parameters
num_epochs = 10
epsilon = 0.03  # perturbation magnitude (8/255 ≈ 0.031):contentReference[oaicite:2]{index=2}
best_acc = 0.0

# Training loop with FGSM adversarial training
for epoch in range(num_epochs):
    model.train()
    # Learning rate schedule (decay at epoch 5)
    if epoch == 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # FGSM Attack generation
        data.requires_grad = True  # needed for computing input gradients:contentReference[oaicite:3]{index=3}
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        data_grad = data.grad.data
        # Apply FGSM perturbation: x_adv = x + epsilon * sign(grad):contentReference[oaicite:4]{index=4}
        perturbed_data = data + epsilon * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1).detach()

        # Train on adversarial examples
        optimizer.zero_grad()
        output_adv = model(perturbed_data)
        loss_adv = criterion(output_adv, target)
        loss_adv.backward()
        optimizer.step()

    # Evaluate clean accuracy each epoch (optional)
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"out/models/{model_name}_best.pt")
    print(f"Epoch {epoch+1}/{num_epochs}, Clean Test Acc: {acc:.4f}")

# Load best model for final evaluation
model.load_state_dict(torch.load(f"out/models/{model_name}_best.pt", map_location=device))
model.eval()

# Evaluate FGSM adversarial accuracy
def eval_fgsm(model, device, loader, epsilon):
    model.eval()
    correct = 0; total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        data_adv = torch.clamp(data + epsilon * data_grad.sign(), 0, 1).detach()
        output_adv = model(data_adv)
        pred = output_adv.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / total

# Evaluate PGD adversarial accuracy
def eval_pgd(model, device, loader, epsilon, alpha=0.007, num_iter=5):
    model.eval()
    correct = 0; total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        delta = torch.zeros_like(data, requires_grad=True)
        for _ in range(num_iter):
            output = model(data + delta)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            grad = delta.grad.data
            delta.data = (delta.data + alpha * grad.sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        data_adv = torch.clamp(data + delta, 0, 1).detach()
        output_adv = model(data_adv)
        pred = output_adv.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / total

# Compute final accuracies
# Clean accuracy
with torch.no_grad():
    correct = 0; total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    clean_acc = correct / total

# FGSM adversarial accuracy
fgsm_acc = eval_fgsm(model, device, test_loader, epsilon)
# PGD adversarial accuracy
pgd_acc = eval_pgd(model, device, test_loader, epsilon, alpha=0.007, num_iter=5)

print(f"Final Clean Accuracy: {clean_acc:.4f}")
print(f"FGSM (epsilon={epsilon}) Adversarial Accuracy: {fgsm_acc:.4f}")
print(f"PGD (epsilon={epsilon}, iter=5) Adversarial Accuracy: {pgd_acc:.4f}")

# Save final best model for submission
model_path = f"out/models/{model_name}_best.pt"
torch.save(model.state_dict(), model_path)

# Submit model to evaluation server (replace TOKEN with your assignment token)
try:
    response = requests.post(
        "http://34.122.51.94:9090/robustness",
        files={"file": open(model_path, "rb")},
        headers={"token": "55172888", "model-name": model_name}
    )
    print("Server response:", response.json())
except Exception as e:
    print(f"Submission failed: {e}")
