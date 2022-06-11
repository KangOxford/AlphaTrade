import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ------------------------------------------------ #
#                                                  #
#    ACGAN May.2022                                #
#    Written by Kang                               #
#    Funtion:                                      #
#      For the network architecture of the ACGAN   #
#    Dependency:                                   #
#      Argument_Parser in ArgumentParser           #
#                                                  #
# ------------------------------------------------ #

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            self.block(),
            self.block(),
            self.block(),
            self.block()
        )
        self.option = opt 
    def block(self,in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels, 0.8), nn.LeakyReLU(0.01)
        )
    def forward(self, input):
        return self.discriminator(input)

class Generator(nn.Module):
    def __init__(self, noise, latent_dim, generated_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self.block(noise, generated_dim*16,4,1,0),
            self.block(generated_dim*16, generated_dim*8,4,2,1),
            self.block(generated_dim*8, generated_dim*4,4,2,1),
            self.block(generated_dim*4, generated_dim*2,2,2,1),
            nn.ConvTranspose2d(generated_dim*2,latent_dim,4,2,1),nn.Tanh()
        )
    def block(self,in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.LeakyReLU(0.01)
        )
    def forward(self, input_lanten, input_generate):
        return self.generator(input_lanten, input_generate)