import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def initializing_weights(model):
    for item in model.modules():
        if isinstance(item, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(item.weight.data, 0.0, 0.01)
