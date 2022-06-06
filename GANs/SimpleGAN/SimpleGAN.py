import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, input):
        super.__init__()
        self.discriminator = nn.Sequential()
        self.discriminator.add(nn.Linear(input, 256))
        self.discriminator.add(nn.LeakyReLU(0.01))
        self.discriminator.add(nn.Linear(256,1))
        self.discriminator.add(nn.Tanh())
    def forward(self, input):
        return self.discriminator(input)

class Generator(nn.Module):
    def __init__(self, latent_dim, generated_dim):
        super.__init__()
        
