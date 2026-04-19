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
    # 1. Reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cpu")
    os.makedirs("out/models", exist_ok=True)

    # 2. Data transforms
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    # 3. Model setup
    model_name = "resnet34"
    allowed_models = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
    }
    model = allowed_models[model_name](weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 22], gamma=0.1)

    # 4. FGSM attack
    def fgsm_attack(model, data, target, eps):
        data_adv = data.clone().detach().requires_grad_(True)
        output = model(data_adv)
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data_adv.grad.data
        perturbed_data = torch.clamp(data_adv + eps * data_grad.sign(), 0, 1)
        return perturbed_data.detach()

    # 5. Training loop (adversarial training)
    epsilon = 8 / 255
    num_epochs = 25
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Create adversarial samples
            data_adv = fgsm_attack(model, data, target, epsilon)

            # Train on adversarial examples
            optimizer.zero_grad()
            output = model(data_adv)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate clean accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs} - Clean Accuracy: {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"out/models/{model_name}_best.pt")

    # 6. Evaluation (FGSM & PGD)
    def eval_fgsm(model, loader, eps):
        model.eval()
        correct = 0
        total = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data_adv = fgsm_attack(model, data, target, eps)
            output = model(data_adv)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        return correct / total

    def eval_pgd(model, loader, eps, alpha=2/255, steps=7):
        model.eval()
        correct = 0
        total = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            delta = torch.zeros_like(data, requires_grad=True)
            for _ in range(steps):
                output = model(data + delta)
                loss = criterion(output, target)
                model.zero_grad()
                loss.backward()
                grad = delta.grad.detach()
                delta.data = (delta + alpha * grad.sign()).clamp(-eps, eps)
                delta.grad.zero_()
            adv_data = torch.clamp(data + delta, 0, 1).detach()
            output = model(adv_data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        return correct / total

    # Load best model
    model.load_state_dict(torch.load(f"out/models/{model_name}_best.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    clean_acc = correct / total

    fgsm_acc = eval_fgsm(model, test_loader, epsilon)
    pgd_acc = eval_pgd(model, test_loader, epsilon)

    print(f"\nFinal Clean Accuracy: {clean_acc*100:.2f}%")
    print(f"FGSM Accuracy: {fgsm_acc*100:.2f}%")
    print(f"PGD Accuracy: {pgd_acc*100:.2f}%")

    # 7. Save and Submit
    final_path = "out/models/dummy_submission.pt"
    torch.save(model.state_dict(), final_path)

    # Validate format
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

    # Submit (Replace TOKEN below)
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
