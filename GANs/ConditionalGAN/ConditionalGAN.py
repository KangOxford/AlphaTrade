import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ----------------------------------------------------------- #
#                                                             #
#    Conditional GAN May.2022                                 #
#    Written by Kang                                          #
#    Funtion:                                                 #
#      For the network architecture of the Conditional GAN    #
#    Dependency:                                              #
#      Argument_Parser in ArgumentParser                      #
#                                                             #
#                                                ┌────┐       # 
#    ┌─────────┐                                 │real│       # 
#    │Z (noise)├───┐                           ┌►│fake│       # 
#    └─────────┘   │   ┌─┐   ┌──────┐          │ └────┘       # 
#                  ├──►│G├──►│X_fake├──┐       │              #
#    ┌─────────┐   │   └─┘   └──────┘  │   ┌─┐ │ ┌──────┐     # 
#    │C (class)├───┘                   ├──►│D├─┤ │Class1│     #  
#    └─────────┘             ┌──────┐  │   └─┘ └►│...   │     #
#                            │X_real├──┘         │ClassN│     #
#                            └──────┘            └──────┘     #
# ----------------------------------------------------------- #

class Discriminator(nn.Module):
    def __init__(self, noise, labels, options):
        super(Generator, self).__init__()
        self.label_embedding    = nn.Embedding(options.number_classes, options.number_classes),
        self.view_generated_dim = int(options.channel_dim * options.generated_dim * options.generated_dim)
        self.input_dimension    = options.number_classes + self.view_generated_dim
        self.hidden_layer_dim   = options.hidden_layer_dim
        self.discriminator      = nn.Sequential(
            self.block(self.input_dimension, self.hidden_layer_dim * 4),
            self.block(self.hidden_layer_dim * 4, self.hidden_layer_dim * 4),
            self.block(self.hidden_layer_dim * 4, self.hidden_layer_dim * 4),
            nn.Linear(self.hidden_layer_dim  * 4, 1)  , nn.Tanh()
        )
    def block(self,in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Dropout(0.4), nn.LeakyReLU(0.01)
        )
    def forward(self, generated, labels):
        discriminator = self.discriminator(torch.cat((
            generated.view(generated.size(0), -1), 
            self.label_embedding(labels)
            ), -1))
        return discriminator


class Generator(nn.Module):
    def __init__(self, options):
        super(Generator, self).__init__()
        self.label_embedding    = nn.Embedding(options.number_classes, options.number_classes),
        self.input_dimension    = options.lantet_dim + options.number_classes
        self.hidden_layer_dim   = options.hidden_layer_dim
        self.generated_shape    = (options.channel_dim, options.generated_dim, options.generated_dim)
        self.view_generated_dim = int(options.channel_dim * options.generated_dim * options.generated_dim)
        self.generator          = nn.Sequential(
            self.block(self.input_dimension, self.hidden_layer_dim),
            self.block(self.hidden_layer_dim * 1, self.hidden_layer_dim * 2),
            self.block(self.hidden_layer_dim * 2, self.hidden_layer_dim * 4),
            self.block(self.hidden_layer_dim * 4, self.hidden_layer_dim * 8),
            nn.Linear(self.hidden_layer_dim  * 8, self.view_generated_dim)  , nn.Tanh()
        )
    def block(self,in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels, 0.8), nn.LeakyReLU(0.01)
        )
    def forward(self, noise, labels):
        generated = self.generator(torch.cat((self.label_embedding(labels), noise),-1))
        return generated.view(generated.size(0), *self.generated_shape)





