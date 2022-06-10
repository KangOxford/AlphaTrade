from configparser import Interpolation
from dis import dis
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def gradient_penalty(discriminator, real, fake):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # >>> 01 Iintailize Interpolation <<<
    alpha = torch.rand((real.shape[0],1,1,1)).repeat(1,real.shape[1],real.shape[2],real.shape[3]).to(device)
    Interpolation = alpha * real + (1 - alpha) * fake 
    # >>> 02 Interpolation scores of the discriminator  <<<
    scores = discriminator(Interpolation)
    # >>> 03 Computing gradients of the scores  <<<
    gradients = torch.autograd.grad(Interpolation, scores, torch.ones_like(scores), create_graph = True, retain_graph = True)
    gradients_norm = gradients.view(gradients.shape[0], -1).norm(2, dim = 1)
    return torch.mean((gradients_norm - 1)**2)