from torchvision.models import resnet18, resnet34, resnet50
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Resnet:
    def __init__(self, architecture, embedding_dim):
        self.architecture = architecture
        self.emb_dim = embedding_dim

    def get_model(self):
        if self.architecture == "resnet18":
            backbone = resnet18(pretrained=False)
        elif self.architecture == "resnet34":
            backbone = resnet34(pretrained=False)
        elif self.architecture == "resnet50":
            backbone = resnet50(pretrained=False)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        backbone.fc = nn.Linear(backbone.fc.in_features, self.emb_dim)
        return backbone


class BuildModel:
    def __init__(self):
        pass

    def build_surrogate_model(self, arch: str, embedding_dim: int):
        """Build surrogate model architecture"""
        if arch == "resnet18":
            return Resnet("resnet18", embedding_dim).get_model()
        elif arch == "resnet34":
            return Resnet("resnet34", embedding_dim).get_model()
        elif arch == "resnet50":
            return Resnet("resnet50", embedding_dim).get_model()
        elif arch == "cnn":
            return SimpleCNN(embedding_dim)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
