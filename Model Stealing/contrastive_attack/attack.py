import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from PIL import Image
from model import BuildModel
from typing import Dict, List, Tuple, Optional
from dataset import QueryDataset
from loss import LossFunctions
import pdb


class APIModelStealer:
    def __init__(
        self,
        surrogate_arch: str = "resnet34",
        embedding_dim: int = 1024,
        loss_type: str = "infonce",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):

        self.device = device
        self.loss_type = loss_type
        self.embedding_dim = embedding_dim

        # Initialize surrogate model architecture
        self.surrogate_model = BuildModel().build_surrogate_model(
            surrogate_arch, embedding_dim
        )
        self.surrogate_model.to(device)

        # Initialize loss function
        self.lossobj = LossFunctions()
        self.criterion = self.lossobj.get_loss_function(loss_type)
        self.total_queries = 0
        self.query_costs = []

    def steal_model(
        self,
        query_dataset,
        embeddings,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.0003,
    ):
        def replace_ids_with_embeddings(x, y):
            embeddings_list = [y[id_.item()] for id_ in x]
            embeddings_array = np.stack(embeddings_list)
            return embeddings_array

        """
        Main model stealing loop using contrastive learning approach
        """

        print(f"Starting model stealing attack...")
        print(f"Loss function: {self.loss_type}")

        # Setup optimizer and data loader
        optimizer = torch.optim.Adam(
            self.surrogate_model.parameters(), lr=learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_size = int(0.75 * len(query_dataset))
        test_size = len(query_dataset) - train_size

        # Split the dataset
        train_dataset, test_dataset = random_split(
            query_dataset, [train_size, test_size]
        )

        query_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.surrogate_model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            queries_this_epoch = 0

            for batch_idx, (images, _, ids) in enumerate(query_loader):
                # Create augmented views for contrastive learning
                if isinstance(images, list):
                    images = torch.cat(images, dim=0)
                    ids = torch.cat(ids, dim=0)

                images = images.to(self.device)

                # Query victim API for embeddings
                with torch.no_grad():
                    victim_embeddings = replace_ids_with_embeddings(ids, embeddings)
                    victim_embeddings = torch.tensor(victim_embeddings)

                # Get surrogate model embeddings
                surrogate_embeddings = self.surrogate_model(images)

                victim_embeddings = victim_embeddings.to(self.device)
                surrogate_embeddings = surrogate_embeddings.to(self.device)

                # Compute loss based on selected strategy
                if self.loss_type == "mse":
                    loss = self.criterion(surrogate_embeddings, victim_embeddings)

                elif self.loss_type == "infonce":
                    # Combine victim and surrogate embeddings for contrastive loss
                    all_embeddings = torch.cat(
                        [surrogate_embeddings, victim_embeddings], dim=0
                    )
                    logits, labels = self.lossobj.info_nce_loss(
                        all_embeddings, batch_size
                    )
                    loss = self.criterion(logits, labels)

                elif self.loss_type == "cosine":
                    # Maximize cosine similarity between embeddings
                    similarity = self.criterion(surrogate_embeddings, victim_embeddings)
                    loss = -similarity.mean()  # Negative for maximization

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                queries_this_epoch += len(images)

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, "
                        f"Loss: {loss.item():.4f}, Queries: {self.total_queries}"
                    )

            scheduler.step()
            avg_loss = epoch_loss / (batch_idx + 1)
            print(
                f"Epoch {epoch+1} completed. Average Training Loss: {avg_loss:.4f}, "
                f"Total Queries: {self.total_queries}"
            )

            distance = self.evaluate_stolen_model(test_loader, embeddings)
            print(
                f"Epoch {epoch+1} completed. Average Validation Distance: {distance:.4f}, "
                f"Total Queries: {self.total_queries}"
            )

        print(f"Model stealing completed!")
        print(f"Total queries used: {self.total_queries}")
        if self.query_costs:
            print(f"Total estimated cost: ${sum(self.query_costs):.2f}")

        return self.surrogate_model

    def evaluate_stolen_model(self, test_dataloader, embeddings):
        def replace_ids_with_embeddings(x, y):
            embeddings_list = [y[id_.item()] for id_ in x]
            embeddings_array = np.stack(embeddings_list)
            return embeddings_array

        self.surrogate_model.eval()
        distances = []
        similarities = []
        for batch_idx, (images, _, ids) in enumerate(test_dataloader):
            with torch.no_grad():
                if isinstance(images, list):
                    images = torch.cat(images, dim=0)
                    ids = torch.cat(ids, dim=0)

                images = images.to(self.device)

                victim_embeddings = replace_ids_with_embeddings(ids, embeddings)
                victim_embeddings = torch.tensor(victim_embeddings)
                surrogate_embeddings = self.surrogate_model(images)

                victim_embeddings = victim_embeddings.to(self.device)
                surrogate_embeddings = surrogate_embeddings.to(self.device)

                # Calculate cosine similarity
                avg_distance = torch.norm(
                    victim_embeddings - surrogate_embeddings, p=2, dim=1
                ).mean()
                distances.append(avg_distance.item())
        return distances.mean()

        avg_similarity = np.mean(similarities)
        print(f"Average cosine similarity with victim: {avg_similarity:.4f}")

        return avg_similarity

    def save_stolen_model(self, filepath: str):
        """Save the stolen model"""
        torch.save(
            self.surrogate_model,
            filepath,
        )
        print(f"Stolen model saved to {filepath}")
