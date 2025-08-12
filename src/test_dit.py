from contextlib import contextmanager

import torch
import torch.nn as nn

from modules import DiT_models


@contextmanager
def default_device(device: str):
    prev = torch._C._get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(prev)


def test_dit_models():
    with default_device("cuda"), torch.no_grad():
        model = DiT_models["DiT-L/2"](input_size=64, in_channels=8)
        x = torch.randn(2, 8, 64, 64)
        t = torch.rand((2,)).float()
        y = torch.randint(0, 1000, (2,))
        output = model(x, t, y)
        assert output.shape == (2, 16, 64, 64)


def get_beta_schedule(num_timesteps, schedule="linear"):
    if schedule == "linear":
        betas = torch.linspace(1e-4, 2e-2, num_timesteps)
    else:
        raise NotImplementedError()
    return betas


def test_diffusion_training_loop():
    with default_device("cuda"):
        # Model
        in_channels = 4
        model = DiT_models["DiT-S/8"](input_size=32, in_channels=in_channels, learn_sigma=True)
        model.train()

        # Diffusion params
        num_timesteps = 1000
        betas = get_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        # Training loop
        for step in range(5):
            optimizer.zero_grad()

            # Dummy data
            x_start = torch.randn(2, in_channels, 32, 32)
            y = torch.randint(0, 1000, (2,))
            t = torch.randint(0, num_timesteps, (x_start.shape[0],))

            # Sample noise and create noisy image
            noise = torch.randn_like(x_start)
            alpha_bar_t = alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)

            x_t = torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1.0 - alpha_bar_t) * noise

            # Forward pass
            output = model(x_t, t.float(), y)

            # We are only interested in predicting the noise (first half of the output).
            predicted_noise = output.chunk(2, dim=1)[0]

            # Loss calculation and backward pass
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            print(f"Step {step}, Loss: {loss.item()}")
            assert not torch.isnan(loss)


if __name__ == "__main__":
    test_dit_models()
    test_diffusion_training_loop()
