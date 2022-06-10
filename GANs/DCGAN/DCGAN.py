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
            self._block(feature,feature*2,4,2,1),
            self._block(feature*2,feature*4,4,2,1),
            self._block(feature*4,feature*8,4,2,1),
            nn.Cov2d(feature*8,1,kernel_size = 4)
        )
    def _block(self,in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.LeakyReLU(0.01)
        )
    def forward(self, input):
        return self.discriminator(input)

class Generator(nn.Module):
    def __init__(self, noise, latent_dim, generated_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self._block(noise, generated_dim*16,4,1,0),
            self._block(generated_dim*16, generated_dim*8,4,2,1),
            self._block(generated_dim*8, generated_dim*4,4,2,1),
            self._block(generated_dim*4, generated_dim*2,2,2,1),
            nn.ConvTranspose2d(generated_dim*2,latent_dim,4,2,1),nn.Tanh()
        )
    def _block(self,in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.LeakyReLU(0.01)
        )
    def forward(self, input_lanten, input_generate):
        return self.generator(input_lanten, input_generate)