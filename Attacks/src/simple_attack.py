import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import Subset, DataLoader
from dataset import MembershipDataset, TaskDataset
from torchvision.models import resnet18
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


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

indices = list(range(len(data)))
split1_indices = [i for i in indices if data[i][3] == 0]
split2_indices = [i for i in indices if data[i][3] == 1]

nonmember_data = Subset(data, split1_indices)
member_data = Subset(data, split2_indices)
dlm = DataLoader(member_data, batch_size=1, shuffle=True)
dlnm = DataLoader(nonmember_data, batch_size=1, shuffle=True)


def extract_attack_features(model, x, device="cuda"):
    model.to(device)
    model.eval()
    for i in range(len(x)):
        x[i] = x[i].to(device)
    x[1].requires_grad = True

    # Forward pass
    logits = model(x[1])
    probs = F.softmax(logits, dim=1)
    confidence, pred_label = probs.max(dim=1)
    pred_label = pred_label.to(torch.float)
    loss = F.cross_entropy(logits, x[2], reduction="none")
    grads = []
    for i in range(len(x[0])):
        model.zero_grad()
        loss[i].backward(retain_graph=True)
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += (param.grad.detach().norm() ** 2).item()
        grads.append(grad_norm**0.5)
    grads = torch.tensor(grads, device=device)
    features = torch.stack([confidence, loss, grads])
    return features.detach().cpu()


def build_attack_dataset(model, member_loader, nonmember_loader, device="cuda"):
    X, y = [], []

    # Members
    for i, xb in enumerate(member_loader):
        if i % 20 == 0:
            print(i)
        feats = extract_attack_features(model, xb, device)
        X.append(feats)
        y.append(torch.ones(len(feats)))

    # Non-members
    for i, xb in enumerate(nonmember_loader):
        if i % 20 == 0:
            print(i)
        feats = extract_attack_features(model, xb, device)
        X.append(feats)
        y.append(torch.zeros(len(feats)))
    return X, y


def build_private_dataset(model, test_loader, device="cuda"):
    X = []
    ids = []

    for i, xb in enumerate(test_loader):
        if i % 20 == 0:
            print(i)
        feats = extract_attack_features(model, xb, device)
        X.append(feats)
        ids.extend(xb[0])
    X = torch.cat(X).numpy()
    return X, ids


X, y = build_attack_dataset(model_target, dlm, dlnm, device="cuda")
X_new = torch.cat(X).numpy()
y_new = torch.cat(y).numpy()

# X and y can be numpy arrays, pandas DataFrames/Series, or PyTorch tensors
X_train, X_val, y_train, y_val = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42, shuffle=True
)


params = {
    "n_estimators": [10, 50, 100, 300, 600],
    "min_child_weight": [50],
    "gamma": [1.5],
    "subsample": [0.8],
    "colsample_bytree": [1.0],
    "max_depth": [3, 10, 50],
    "learning_rate": [0.05],
}


xgb = XGBClassifier(
    nthread=6,
    eval_metric="auc",
    device="cuda:0",
    enable_categorical=True,
    tree_method="hist",
)

cvFold = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
random_search = GridSearchCV(xgb, param_grid=params, n_jobs=10, cv=cvFold)
attack_model = random_search.fit(X_train, y_train)
dlp = DataLoader(data_priv, batch_size=64, shuffle=True)


X_submission, sub_ids = build_private_dataset(model_target, dlp)

pred_prob = attack_model.predict_proba(X_submission)

df = pd.DataFrame(
    {
        "ids": sub_ids,
        "score": pred_prob,
    }
)
df.to_csv("test_simple.csv", index=None)
