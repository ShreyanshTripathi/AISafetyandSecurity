import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from scipy.stats import norm
import torch
from torchvision import transforms
from torchvision.models import resnet18

from torch.utils.data import Dataset, random_split
from dataset import MembershipDataset, TaskDataset
import pandas as pd


def logit_scaling(p):
    return np.log(p / (1 - p))


def train_shadow_model(dataloader_out, epochs):
    model = resnet18(weights="DEFAULT")
    model.fc = torch.nn.Linear(512, 44)
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for s in range(epochs):
        for x in dataloader_out:
            optimizer.zero_grad()
            logits = model(x[1].to("cuda"))
            loss = criterion(logits, x[2].to("cuda"))
            loss.backward()
            optimizer.step()

    return model


def sample_n_points(dataset, N):
    total_len = len(dataset)
    if N > total_len:
        raise ValueError(f"N={N} is larger than dataset size {total_len}")
    indices_out = torch.randperm(total_len)[:N]  # Random unique indices
    out_subset = Subset(dataset, indices_out)
    return out_subset


def logit_scaling(p):
    return np.log(p / (1 - p + 1e-8))


def get_confidence(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        a = probs[torch.arange(probs.size(0)), y]
    return a


def offline_lira_attack(target_model, shadow_models, x_target, y_target):
    # Get target confidence
    target_conf = logit_scaling(get_confidence(target_model, x_target, y_target).cpu())
    # Get shadow model confidences
    shadow_confs = [
        logit_scaling(get_confidence(model.to("cuda"), x_target, y_target).cpu())
        for model in shadow_models
    ]
    # Compute p-value
    mu = np.mean(shadow_confs)
    std = np.std(shadow_confs) + 1e-8  # Prevent division by zero
    p_value = 1 - norm.cdf(target_conf, loc=mu, scale=std)

    return p_value, target_conf, mu, std


mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

model_target = resnet18(pretrained=False)
model_target.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("./01_MIA.pt", map_location="cpu")

model_target.load_state_dict(ckpt)

data: MembershipDataset = torch.load("./pub.pt", weights_only=False)
data_priv: TaskDataset = torch.load("./priv_out.pt", weights_only=False)
train_set_size = int(0.85 * len(data))
val_set_size = len(data) - train_set_size
train_data, val_data = random_split(data, [train_set_size, val_set_size])
dlt = DataLoader(train_data, batch_size=64, shuffle=True)
dlv = DataLoader(val_data, batch_size=64, shuffle=True)
dlp = DataLoader(data_priv, batch_size=512, shuffle=True)

shadow_models = []
for i in range(100):
    print(i)
    out_subset = sample_n_points(train_data, int(0.80 * len(train_data)))
    dataloader_out = DataLoader(out_subset, batch_size=512, shuffle=True)
    model = train_shadow_model(dataloader_out, 20)
    shadow_models.append(model.to("cpu"))

results_val = []
mem_nonmem = []
ids = []
for idx, i in enumerate(dlp):
    if idx % 10 == 0:
        print(idx)
    a = offline_lira_attack(
        model_target.to("cuda"), shadow_models, i[1].to("cuda"), i[2].to("cuda")
    )
    results_val.append(a)
    ids.extend(i[0].cpu())

conf = []
for idx, i in enumerate(results_val):
    conf.extend(i[0])
ids = [i.item() for i in ids]

df = pd.DataFrame(
    {
        "ids": ids,
        "score": conf,
    }
)
df.to_csv("test_lira.csv", index=None)
