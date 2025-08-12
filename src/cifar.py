# %%
import lovely_tensors as lt

lt.monkey_patch()

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributions as D
import torchvision
import torchvision.transforms as TF
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader, Dataset

sns.set_theme(style="dark")

# %%
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# %%
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    # if torch.backends.mps.is_available():
    # return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_default_device()

# %%
from typing import cast

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(DEVICE)
# vae = cast(AutoencoderKL, torch.compile(vae, mode="max-autotune"))
vae.eval()
if DEVICE == torch.device("cuda"):
    # vae = cast(AutoencoderKL, torch.compile(vae, mode="max-autotune"))
    pass

# %%
vae_processor = VaeImageProcessor(do_normalize=True)

# %%
import torchvision.transforms.functional as TVF


def normalize(x: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, vae_processor.normalize(x))


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, vae_processor.denormalize(x))


def resize(x: torch.Tensor) -> torch.Tensor:
    return TVF.resize(x, [256, 256])


transform = TF.Compose(
    [
        TF.ToTensor(),
        TF.RandomHorizontalFlip(),
        TF.Lambda(lambda x: normalize(x)),
    ]
)


# %%
class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool = True):
        self.cifar = torchvision.datasets.CIFAR10(
            root="/tmp/cifar", download=True, train=train, transform=transform
        )

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx: int):
        x, y = self.cifar[idx]
        t = torch.rand(1)
        return x, y, t


# %%
cifar_train = CifarDataset(train=True)
cifar_test = CifarDataset(train=False)

BATCH_SIZE = 16

train_dataloader = torch.utils.data.DataLoader(
    cifar_train, batch_size=BATCH_SIZE, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    cifar_test, batch_size=BATCH_SIZE, shuffle=True
)


x = resize(next(iter(train_dataloader))[0].to(DEVICE))
latent_shape = vae.encode(x, return_dict=False)[0].mean[0].shape

(
    next(iter(train_dataloader))[0],
    next(iter(train_dataloader))[1],
    next(iter(train_dataloader))[2],
    latent_shape,
)

# %%
from modules import DiT

model = DiT(
    d_model=768,
    patch_size=4,
    img_size=(32, 32),
    n_heads=12,
    n_layers=28,
    in_channels=4,
)

# %%
N_EPOCHS = 200


model = model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

ckpt_path = f"../ckpt/{DiT.__name__}"
if ckpt_path:
    model.load_state_dict(torch.load(f"{ckpt_path}/model.pth", map_location=DEVICE))
    optimizer.load_state_dict(
        torch.load(f"{ckpt_path}/optimizer.pth", map_location=DEVICE)
    )
    scheduler.load_state_dict(
        torch.load(f"{ckpt_path}/scheduler.pth", map_location=DEVICE)
    )

if DEVICE == torch.device("cuda:1"):
    pass
    # model = cast(DiT, torch.compile(model, mode="max-autotune"))

# %%
print(f"Param count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# %%
from torchdiffeq import odeint


@torch.no_grad()
def sample_with_ode(
    model: nn.Module,
    n_samples: int = 500,
    n_steps: int = 25,
):
    model.eval()
    model_device = next(model.parameters()).device
    initial_samples = torch.randn((n_samples, *latent_shape), device=model_device)
    t_span = torch.linspace(0.0, 1.0, n_steps).to(model_device)
    trajectory = odeint(
        model.ode_forward,
        initial_samples,
        t_span,
        method="euler",
        atol=1e-5,
        rtol=1e-5,
    )
    trajectory = cast(torch.Tensor, trajectory)

    return trajectory[-1]


# %%
x = vae.decode(
    vae.encode(next(iter(train_dataloader))[0].to(DEVICE)[:1]).latent_dist.mean
)
print(x)
del x

# %%
import os

from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import resize


def to_latent(vae: AutoencoderKL, x_raw: torch.Tensor) -> torch.Tensor:
    x_raw = resize(x_raw, [256, 256])
    return vae.encode(x_raw, return_dict=False)[0].mean


def from_latent(vae: AutoencoderKL, z: torch.Tensor) -> torch.Tensor:
    return vae.decode(cast(torch.FloatTensor, z), return_dict=False)[0]


# %%
x = denormalize(
    from_latent(vae, to_latent(vae, next(iter(train_dataloader))[0].to(DEVICE)[:1]))
)
print(x)
del x

# %%
from tqdm.auto import tqdm


def train(
    model: nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    verbose: bool = False,
    contrastive_flow_weight: float = 0.00,
):
    def step(x_raw, t):
        x = to_latent(vae, x_raw)
        noise = torch.randn_like(x, device=DEVICE)
        t_expanded = t.unsqueeze(-1).unsqueeze(-1)
        z = t_expanded * x + (1 - t_expanded) * noise
        target_u = x - noise
        u = model(z, t.squeeze(-1))
        loss = F.mse_loss(u, target_u)
        if contrastive_flow_weight > 0.0:
            u_hat = torch.roll(u, shifts=1, dims=0)
            loss_contrastive = F.mse_loss(u, u_hat)
            loss = loss - contrastive_flow_weight * loss_contrastive
        return loss

    @torch.no_grad()
    def get_fid(nfe: int = 25):
        fid = FrechetInceptionDistance().to(DEVICE)
        for i, (val_imgs, _, _) in enumerate(val_dataloader):
            real_images = val_imgs
            real_images = denormalize(real_images).clamp(0, 1)
            real_images = (
                (real_images * 255).to(torch.uint8).to(DEVICE)
            )  # BS x 3 x 32 x 32
            real_images = resize(real_images, [256, 256])
            z_s = sample_with_ode(model, n_samples=real_images.shape[0], n_steps=nfe)
            # z_s = to_latent(vae, val_imgs.to(DEVICE))
            generated_images = from_latent(vae, z_s)
            generated_images = denormalize(generated_images).clamp(0, 1)
            generated_images = (
                (generated_images * 255).to(torch.uint8).to(DEVICE)
            )  # BS x 3 x 32 x 32
            fid.update(real_images, real=True)
            fid.update(generated_images, real=False)
            if i * BATCH_SIZE > 10000:
                break
        return fid.compute()

    def ckpt_callback(epoch: int, val_loss: float):
        ckpt_dir = "../ckpt"
        subdir = f"{ckpt_dir}/{DiT.__name__}2"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        torch.save(model.state_dict(), f"{subdir}/model.pth")
        torch.save(optimizer.state_dict(), f"{subdir}/optimizer.pth")
        torch.save(scheduler.state_dict(), f"{subdir}/scheduler.pth")
        print(f"Saved checkpoint at epoch {epoch} with val loss {val_loss:.4f}")

    log_interval = 1
    model.train()
    best_val_loss = float("inf")
    loss_history = []
    val_loss_history = []
    fid_25_history = []
    fid_50_history = []
    try:
        for epoch in range(n_epochs):
            losses = []
            for x_raw, _, t in tqdm(dataloader):
                loss = step(x_raw.to(DEVICE), t.to(DEVICE))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            scheduler.step()

            val_losses = []
            with torch.no_grad():
                for x_raw, _, t in val_dataloader:
                    loss = step(x_raw.to(DEVICE), t.to(DEVICE))
                    val_losses.append(loss.item())

            current_loss = float(np.mean(losses))
            current_val_loss = float(np.mean(val_losses))

            fid_25 = get_fid(nfe=25).cpu().numpy()
            fid_50 = get_fid(nfe=50).cpu().numpy()
            fid_25_history.append(fid_25)
            fid_50_history.append(fid_50)
            if (epoch % log_interval == 0 or epoch == n_epochs - 1) and verbose:
                print(
                    f"Epoch {epoch}\t loss: {current_loss:.4f}\t val loss: {current_val_loss:.4f}\t FID_25: {fid_25:.4f}\t FID_50: {fid_50:.4f}"
                )

            loss_history.append(current_loss)
            val_loss_history.append(current_val_loss)

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                ckpt_callback(epoch, current_val_loss)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    train_dict = {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "fid_25_history": fid_25_history,
        "fid_50_history": fid_50_history,
    }
    return train_dict


# %%
train_dict = train(
    model=model,
    dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    n_epochs=N_EPOCHS,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True,
)

# %%
plt.plot(train_dict["loss_history"])
plt.plot(train_dict["val_loss_history"])
plt.legend(["train", "val"])
plt.show()

plt.plot(train_dict["fid_25_history"])
plt.plot(train_dict["fid_50_history"])
plt.legend(["FID_25", "FID_50"])
plt.show()

# %%
z_s = sample_with_ode(model, n_samples=4, n_steps=25)
generated_images = from_latent(vae, z_s)
generated_images = (generated_images * 255).to(torch.uint8).to(DEVICE)

# %%
import torchvision.transforms as TF

TF.ToPILImage()(generated_images[2]).resize((128, 128))

# %%
TF.ToPILImage()(next(iter(test_dataloader))[0][0]).resize((128, 128))

# %%


# %%
device_2 = torch.device("cuda:0")

vae = vae.to(device_2)

fid = FrechetInceptionDistance().to(device_2)

for i, (real, _, _) in enumerate(
    torch.utils.data.DataLoader(cifar_test, batch_size=4, shuffle=True)
):
    if i * 4 > 512:
        break
    real_images = real.to(device_2)

    real_images = resize(real_images)
    generated_images = resize(generated_images)
    generated_images = vae.decode(vae.encode(real_images).latent_dist.mean).sample

    real_images = denormalize(real_images)
    generated_images = denormalize(generated_images)

    real_images = (real_images * 255).to(torch.uint8).to(device_2)
    generated_images = (generated_images * 255).to(torch.uint8).to(device_2)

    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)

fid.compute()

# %%


# %%
