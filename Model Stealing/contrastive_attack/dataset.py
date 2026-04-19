import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torchvision.utils as vutils


from typing import Tuple
import random
from collections import defaultdict


class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class QueryDataset(Dataset):
    def __init__(self, dataset):
        transform0 = v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )
        transform1 = v2.Compose(
            [
                v2.RandomVerticalFlip(1.0),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        transform2 = v2.Compose(
            [
                v2.RandomHorizontalFlip(1.0),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        transform3 = v2.Compose(
            [
                v2.RandomInvert(1.0),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        transform4 = v2.Compose(
            [
                v2.RandomRotation(90),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        transform5 = v2.Compose(
            [
                v2.RandomAffine(
                    degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                ),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        self.transforms_list = [
            transform0,
            transform1,
            transform2,
            transform3,
            transform4,
            transform5,
        ]

        self.dataset = dataset
        self.selected_imgs, self.selected_labels, self.selected_ids = (
            self.dataset.imgs,
            self.dataset.labels,
            self.dataset.ids,
        )

    def __init__(self, dataset, class_ratio):
        transform0 = v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )
        transform1 = v2.Compose(
            [
                v2.RandomVerticalFlip(1.0),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        transform2 = v2.Compose(
            [
                v2.RandomHorizontalFlip(1.0),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        transform3 = v2.Compose(
            [
                v2.RandomInvert(1.0),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        transform4 = v2.Compose(
            [
                v2.RandomRotation(90),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        transform5 = v2.Compose(
            [
                v2.RandomAffine(
                    degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                ),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                ),
            ]
        )

        self.transforms_list = [
            transform0,
            transform1,
            transform2,
            transform3,
            transform4,
            transform5,
        ]

        self.dataset = dataset
        self.selected_imgs, self.selected_labels, self.selected_ids = (
            self.filter_dataset(class_ratio)
        )

    def select_by_indices(self, indices, A, B, C):
        selected_A = [A[i] for i in indices]
        selected_B = [B[i] for i in indices]
        selected_C = [C[i] for i in indices]
        return selected_A, selected_B, selected_C

    def sample_indices_by_class(self, class_ids, n_samples_per_class, seed=42):
        if seed is not None:
            random.seed(seed)

        # Map each class to its indices
        class_to_indices = defaultdict(list)
        for idx, class_id in enumerate(class_ids):
            class_to_indices[class_id].append(idx)

        # Collect samples
        sampled_indices = []
        for class_id, n in n_samples_per_class.items():
            if class_id not in class_to_indices:
                raise ValueError(f"Class {class_id} not found in class_ids")
            if len(class_to_indices[class_id]) < n:
                raise ValueError(
                    f"Class {class_id} has only {len(class_to_indices[class_id])} samples, requested {n}"
                )

            sampled_indices.extend(random.sample(class_to_indices[class_id], n))

        return sampled_indices

    def filter_dataset(self, class_ratio):
        images = self.dataset.imgs
        labels = self.dataset.labels
        labels = [int(label) for label in labels]
        ids = self.dataset.ids
        sampled_indices = self.sample_indices_by_class(labels, class_ratio)
        self.selected_imgs, self.selected_labels, self.selected_ids = (
            self.select_by_indices(sampled_indices, images, labels, ids)
        )
        return self.selected_imgs, self.selected_labels, self.selected_ids

    def __len__(self):
        return len(self.selected_imgs)

    def __getitem__(self, idx):
        image = self.selected_imgs[idx]
        label = self.selected_labels[idx]
        id = self.selected_ids[idx]
        image_list = []
        if len(self.transforms_list) > 0:
            image_list.extend([i(image) for i in self.transforms_list])
        return (
            image_list,
            [label] * (len(self.transforms_list)),
            [id] * (len(self.transforms_list)),
        )


def show_images(tensor_images, nrow=6):
    # Make a grid of images
    grid_img = vutils.make_grid(tensor_images, nrow=nrow, padding=2)
    # Convert to numpy for matplotlib
    np_img = grid_img.numpy()
    # Transpose to HWC (height, width, channels)
    np_img = np_img.transpose((1, 2, 0))
    # Plot
    plt.figure(figsize=(12, 4))
    plt.imshow(np_img)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    class_ratio = {
        40019202: 34,
        10964686: 27,
        62225416: 177,
        26466207: 83,
        24683694: 42,
        90386222: 113,
        15103694: 134,
        14119474: 50,
        75986657: 32,
        62014378: 158,
        15256249: 31,
        64963739: 27,
        83450130: 42,
        92257871: 26,
        50011542: 32,
        68395620: 28,
        19888903: 15,
        31527279: 36,
        67716412: 11,
        23255817: 17,
        74777746: 72,
        75652382: 44,
        55364435: 27,
        33633783: 26,
        70483417: 37,
        75477331: 18,
        23659574: 37,
        65142471: 66,
        6661858: 52,
        12076452: 16,
        43338066: 78,
        34183981: 10,
        98460345: 10,
        32036533: 27,
        85864573: 16,
        47140557: 17,
        93429558: 41,
        77329636: 32,
        91097190: 12,
        9410583: 15,
        75313326: 14,
        78173080: 16,
        37206695: 32,
        15258297: 13,
        20200178: 9,
        47511766: 3,
        80504125: 19,
        79051840: 9,
        40835903: 4,
        32369775: 13,
    }

    dataset = torch.load("../ModelStealingPub.pt", weights_only=False)
    query_dataset = QueryDataset(dataset=dataset, class_ratio=class_ratio)
    print(len(query_dataset))

    # for idx, i in enumerate(query_dataset):
    #     if idx == 0:
    #         continue
    #     show_images(i[0])
    #     break
