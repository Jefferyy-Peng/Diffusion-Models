import argparse
import logging
import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules import UNet
from utils import save_images


class ForwardDiffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, image_size=256, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.device = device

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)

    def sample_timestamps(self, n):
        return torch.randint(1, self.noise_steps, (n,)).to(self.device)

    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ


    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            z = torch.randn(n, 3, self.image_size, self.image_size).to(self.device)
            for i in tqdm(reversed(range(self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(z, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(z)
                else:
                    noise = torch.zeros_like(z)
                z = 1 / torch.sqrt(alpha) * (
                            z - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        z = (z.clamp(-1, 1) + 1) / 2
        z = (z * 255).type(torch.uint8)
        return z

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)), # crop to this range and resample to image_size
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def train(args):
    diffusion = ForwardDiffusion(image_size=args.image_size, device=args.device)
    model = UNet()
    device = args.device
    model.to(device)
    dataloader = get_data(args)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (image, _) in enumerate(pbar):
            image = image.to(device)
            t = diffusion.sample_timestamps(image.shape[0])
            x_t, noise = diffusion.noise_image(image, t)
            noise_pred = model(x_t, t)
            loss = criterion(noise, noise_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=dataloader.batch_size)
        if not os.path.exists(os.path.join("results", args.run_name)):
            os.makedirs(os.path.join("results", args.run_name))
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        if not os.path.exists(os.path.join("models", args.run_name)):
            os.makedirs(os.path.join("models", args.run_name))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"D:\Dataset\Landscape"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)