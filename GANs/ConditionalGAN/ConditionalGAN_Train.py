import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ArgumentParser import Argument_Parser
from ConditionalGAN import Generator, Discriminator

# -------------------------------------------------------- #
#                                                          #
#    Conditional GAN May.2022                              #
#    Written by Kang                                       #
#    Funtion:                                              #
#      For the training process of the Conditional GAN     #
#    Dependency:                                           #
#      Discriminator, Generator in ConditionalGAN          #
#                                                          #
# -------------------------------------------------------- #

# >>> 01 Iintailize ArgumentParser <<<
option = Argument_Parser()

# >>> 02 Iintailize Generator and Discriminator <<<
def initializing_weights(model):
    nn.init.normal_(model.weight.data, 0.0, 0.01)


# discriminator = Discriminator(channels_dim, discriminator_dim).to(device)
# generator = Generator(latent_dim, channels_dim, generated_dim).to(device)
# initializing_weights(discriminator)
# initializing_weights(generator)





