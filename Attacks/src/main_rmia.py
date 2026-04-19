import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from scipy.stats import norm
import torch
from torchvision import transforms
from torchvision.models import resnet18

from torch.utils.data import Dataset, random_split
from dataset import MembershipDataset, TaskDataset


def get_confidence(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        a = probs[torch.arange(probs.size(0)), y]
    return a


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


def rmia(shadow_models, x, y, Ratio_z_list):
    gamma = 2
    conf_ref = torch.Tensor(np.array([0] * x.shape[0])).to("cuda")
    for model in shadow_models:
        conf_ref += get_confidence(model.to("cuda"), x.to("cuda"), y.to("cuda"))
    conf_x_out = conf_ref / len(shadow_models)
    a = 0.2
    conf_x = 0.5 * ((1 + a) * conf_x_out + (1 - a))
    conf_x_theta = get_confidence(model_target.to("cuda"), x.to("cuda"), y.to("cuda"))
    Ratio_x = conf_x_theta / conf_x
    res_list = []
    for i in Ratio_x:
        c = 0
        for j in Ratio_z_list:
            if i / j > gamma:
                c += 1
        res_list.append(c)
    return np.array(res_list) / len(Ratio_z_list)


def get_shadow_models_conf(shadow_models, z):
    conf_ref = torch.Tensor(np.array([0] * z[0].shape[0])).to("cuda")
    for model in shadow_models:
        conf_ref += get_confidence(model.to("cuda"), z[1].to("cuda"), z[2].to("cuda"))
    return conf_ref


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

Ratio_z = []
for z in dlv:
    conf_ref = get_shadow_models_conf(shadow_models, z)
    conf_z = conf_ref / len(shadow_models)
    conf_z_theta = get_confidence(
        model_target.to("cuda"), z[1].to("cuda"), z[2].to("cuda")
    )
    Ratio_z.extend(conf_z_theta / conf_z)
res_idx = []
res_val = []
for idx, i in enumerate(dlt):
    print(idx)
    res_idx.extend(i[0])
    res = rmia(shadow_models, i[1], i[2], Ratio_z)
    res_val.extend(res)

res_idx = [i.item() for i in res_idx]

df = pd.DataFrame(
    {
        "ids": res_idx,
        "score": res_val,
    }
)
df.to_csv("test_rmia.csv", index=None)
