import torch
import os

from DDPM import ForwardDiffusion
from modules import UNet
from utils import plot_images

run_name = "DDPM_Uncondtional"

state_dict = torch.load(os.path.join("models", run_name, f"ckpt.pt"))
model = UNet()

model.load_state_dict(state_dict)
diffuse = ForwardDiffusion(image_size=128, device="cuda")
model.to('cuda')

sampled_images = diffuse.sample(model, n=4)
plot_images(sampled_images)