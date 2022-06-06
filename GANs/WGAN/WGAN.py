from turtle import forward
from regex import F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img, feature):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(img,feature,kernel_size=4,stride=2,padding=1), nn.LeakyReLU(0.01),
            self._block(feature,feature*2,4,2,1)
        )
    def _block(self,in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        )
    def forward(self, input):
        return self.discriminator(input)

class Generator(nn.Module):
    def __init__(self, latent_dim, generated_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),    nn.LeakyReLU(0.01),
            nn.Linear(256, generated_dim), nn.Tanh()
        )
    def forward(self, input_lanten, input_generate):
        return self.generator(input_lanten, input_generate)