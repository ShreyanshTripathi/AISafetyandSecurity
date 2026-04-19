import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunctions:
    def __init__(self):
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def get_loss_function(self, loss_type: str):
        """Initialize loss function for model stealing"""
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "infonce":
            return nn.CrossEntropyLoss()
        elif loss_type == "cosine":
            return nn.CosineSimilarity(dim=1)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def info_nce_loss(self, features: torch.Tensor, batch_size: int):
        """InfoNCE loss for contrastive learning"""
        n = int(features.size()[0] / batch_size)
        labels = torch.cat([torch.arange(batch_size) for i in range(n)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # Mask out diagonal entries
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # Get positive and negative pairs
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / 0.07  # temperature

        return logits, labels
