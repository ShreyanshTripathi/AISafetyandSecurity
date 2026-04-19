import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import requests

def main():
    # Reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cpu")
    os.makedirs("out/models", exist_ok=True)

    # CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training: normalize & augment
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Test: only ToTensor (as per server expectations)
    test_transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    # Model
    model_name = "resnet18"
    allowed_models = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
    }
    model = allowed_models[model_name](weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    num_epochs = 20
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval on unnormalized test set
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
        acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Clean Accuracy: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"out/models/{model_name}_best.pt")

    # Load best model
    model.load_state_dict(torch.load(f"out/models/{model_name}_best.pt", map_location=device))
    model.eval()

    # Save for submission
    final_path = "out/models/dummy_submission.pt"
    torch.save(model.state_dict(), final_path)

    # Format validation
    with open(final_path, "rb") as f:
        try:
            test_model = allowed_models[model_name](weights=None)
            test_model.fc = nn.Linear(test_model.fc.in_features, 10)
            state_dict = torch.load(f, map_location=device)
            test_model.load_state_dict(state_dict, strict=True)
            test_model.eval()
            out = test_model(torch.randn(1, 3, 32, 32))
            assert out.shape == (1, 10)
        except Exception as e:
            raise Exception(f"Model format invalid: {e}")

    # Submit (replace TOKEN)
    try:
        response = requests.post(
            "http://34.122.51.94:9090/robustness",
            files={"file": open(final_path, "rb")},
            headers={"token": "55172888", "model-name": model_name}
        )
        print("Server response:", response.json())
    except Exception as e:
        print("Submission failed:", e)

if __name__ == "__main__":
    main()
