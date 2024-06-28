import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import scipy
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import seaborn as sbs
from pathlib import Path
import os

def mlp_classif(hidden_layer):
    n = len(hidden_layer)
    layers = []
    for i in range(n):
        if i == 0:
            layers.append(nn.Linear(2,hidden_layer[i]))
            layers.append(nn.ReLU())
        elif i == n-1:
            layers.append(nn.Linear(hidden_layer[i],1))
            # layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)

    return model

@torch.no_grad()
def predict(dataloader, model):
    model.eval()
    predictions = np.array([])

    for x_batch, _ in dataloader:
        out = model(x_batch)
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).type(torch.long)
        predictions = np.hstack((predictions, preds.numpy().flatten()))

    predictions = predictions
    return predictions.flatten()

def beta_warping(x, alpha, eps=1e-12):
    return scipy.stats.beta.ppf(x, a=alpha + eps, b=alpha+eps)

def sim_gauss_kernel(dist, tau_max, tau_std):
    dist = dist / np.mean(dist)
    dist_rate = tau_max * np.exp(-(dist - 1) / (2 * tau_std * tau_std))

    return dist_rate

def show_separation(model, X_train, y_train, X_test, y_test, save=False, name_to_save="", score="", model_name="", font_scale=1.):
    plt.rcParams['text.usetex'] = True
    sbs.set_theme(style="white", font_scale=font_scale)

    xx, yy = np.mgrid[-1.5:2.5:.01, -1.:1.5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    batch = torch.from_numpy(grid).type(torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch).reshape(xx.shape))
        probs = probs.numpy().reshape(xx.shape)

    f, ax = plt.subplots(figsize=(16, 10))
    if score:
        # ax.set_title(f"Decision boundary with {model_name} - acc = {score}")
        ax.set_title(f"Accuracy = {score}")
    else:
        ax.set_title(f"Decision boundary with {model_name}")
    contour = ax.contourf(xx, yy, probs, 25, alpha=0.8, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax.scatter(X_train[:,0], X_train[:, 1], c=y_train, s=200,
               cmap=cm_bright, vmin=-.2, vmax=1.2,
               edgecolor="white", linewidths=1.)
    ax.scatter(X_test[:,0], X_test[:, 1], c=y_test, s=200,
               cmap=cm_bright, vmin=-.2, vmax=1.2,
               edgecolor="white", linewidths=1., marker='*')

    ax.set(xlabel="$X_1$", ylabel="$X_2$")
    sbs.despine(left=True, bottom=True)
    if save:
        plt.savefig(name_to_save, dpi=300, bbox_inches='tight')
    else:
        plt.show()

visu_dir = Path("./visu/moons/")
save_figs = True
folder_to_save = "final"

os.makedirs(visu_dir / folder_to_save, exist_ok=True)

X, y = make_moons(n_samples=100, noise=0.15, random_state=0)
classes = np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

X_train_t = torch.from_numpy(X_train).to(torch.float32)
y_train_t = torch.from_numpy(y_train).to(torch.float32)
X_test_t = torch.from_numpy(X_test).to(torch.float32)
y_test_t = torch.from_numpy(y_test).to(torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

hid = (20,20,)

criterion = nn.BCEWithLogitsLoss()

### ERM Training

rng = np.random.RandomState(987)
rng_torch = torch.random.manual_seed(987)

bs = 128
max_epochs = 5000

mlp = mlp_classif(hid)
optim_mlp = torch.optim.Adam(mlp.parameters())

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, generator=rng_torch)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, generator=rng_torch)

for epoch in tqdm(range(max_epochs)):
    for x_batch, y_batch in train_dataloader:
        optim_mlp.zero_grad()
        out = mlp(x_batch)
        loss = criterion(out.flatten(), y_batch)
        loss.backward()
        optim_mlp.step()

score = accuracy_score(y_test, predict(test_dataloader, mlp))
print(score)

show_separation(mlp, X_train, y_train, X_test, y_test, save=save_figs, name_to_save=visu_dir / folder_to_save / "mlp_moons.pdf", score=f"{score:0.3f}", model_name="ERM", font_scale=4.5)

### MIXUP training

rng = np.random.RandomState(987)
rng_torch = torch.random.manual_seed(987)

mixup_alpha = 5.
bs = 128
max_epochs = 5000

mlp_mixup = mlp_classif(hid)
optim_mixup = torch.optim.Adam(mlp_mixup.parameters())

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, generator=rng_torch)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, generator=rng_torch)

for epoch in tqdm(range(max_epochs)):
    for x_batch, y_batch in train_dataloader:
        lamb = rng.beta(mixup_alpha, mixup_alpha)
        index = rng.permutation(len(x_batch))

        x_mix = lamb * x_batch + (1-lamb) * x_batch[index]
        y_mix = lamb * y_batch + (1-lamb) * y_batch[index]
        optim_mixup.zero_grad()
        out = mlp_mixup(x_mix)
        loss = criterion(out.flatten(), y_mix)
        loss.backward()
        optim_mixup.step()

score_mixup = accuracy_score(y_test, predict(test_dataloader, mlp_mixup))
print(score_mixup)

# show_separation(mlp_mixup, X_train, y_train, X_test, y_test, save=save_figs, name_to_save=visu_dir / folder_to_save / f"mlp_moons_mixup{mixup_alpha}.pdf", model_name=f"Mixup with alpha = {mixup_alpha}", score=f"{score_mixup:0.3f}", font_scale=4.5)
show_separation(mlp_mixup, X_train, y_train, X_test, y_test, save=save_figs, name_to_save=visu_dir / folder_to_save / f"mlp_moons_mixup{mixup_alpha}.pdf", model_name="Mixup", score=f"{score_mixup:0.3f}", font_scale=4.5)


### SK MIXUP training

rng = np.random.RandomState(987)
# rng_torch = torch.random.manual_seed(987)

mixup_alpha = 1

tau_max = 5.
tau_std = 0.2

bs = 128
max_epochs = 5000

mlp_warp = mlp_classif(hid)
optim_warp = torch.optim.Adam(mlp_warp.parameters())

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, generator=rng_torch)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, generator=rng_torch)

for epoch in tqdm(range(max_epochs)):
    for x_batch, y_batch in train_dataloader:
        lamb = rng.beta(mixup_alpha, mixup_alpha, x_batch.size(0))
        index = rng.permutation(len(x_batch))

        dist = torch.sqrt(torch.sum((x_batch - x_batch[index])**2, dim=-1)).numpy()
        warp_param = sim_gauss_kernel(dist, tau_max, tau_std)
        k_lamb = torch.tensor(beta_warping(lamb, warp_param)).view(-1,1).float()

        x_mix = k_lamb * x_batch + (1-k_lamb) * x_batch[index]
        y_mix = k_lamb.squeeze(-1) * y_batch + (1-k_lamb.squeeze(-1)) * y_batch[index]
        optim_warp.zero_grad()
        out = mlp_warp(x_mix)
        loss = criterion(out.flatten(), y_mix)
        loss.backward()
        optim_warp.step()

score_warp = accuracy_score(y_test, predict(test_dataloader, mlp_warp))
print(score_warp)

show_separation(mlp_warp, X_train, y_train, X_test, y_test, save=save_figs, name_to_save= visu_dir / folder_to_save / f"mlp_moons_warp_{tau_max}_{tau_std}.pdf", model_name="Warping Mixup", score=f"{score_warp:0.3f}", font_scale=4.5)
