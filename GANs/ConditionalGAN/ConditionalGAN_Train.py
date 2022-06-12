import argparse
import numpy as np

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
options = Argument_Parser()
device = "cuda" if torch.cuda.is_available() else "cpu"

# >>> 02 Iintailize Generator and Discriminator <<<
def initializing_weights(model):
    nn.init.normal_(model.weight.data, 0.0, 0.01)
discriminator = Discriminator(options).to(device)
generator = Generator(options).to(device)
initializing_weights(discriminator)
initializing_weights(generator)

# >>> 03 Initailize Dataset <<<
transforms = transforms.Compose([
    transforms.Resize(options.generated_dim),
    transforms.ToTensor,
    transforms.Normalize([0.1307],[0.3081])
    ])
dataset = datasets.MNIST(root="./dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)

# >>> 04 Iintailize Optimizer <<<
optim_Adam_betas = (options.beta_first_moment, options.beta_second_moment)
optim_discriminator =  optim.Adam(discriminator.parameters(), lr = options.learning_rate, betas=optim_Adam_betas)
optim_generator = optim.Adam(generator.parameters(), lr = options.learning_rate, betas=optim_Adam_betas)
optim_loss = torch.nnMSELoss().to(device)

# >>> 05 Set Tensorboard <<<
writer_gene = SummaryWriter(f"logs/gene")
writer_data = SummaryWriter(f"logs/data")
step = 0 


# ---------------------------------------------- #
# ***************** Training ******************* # 
# ---------------------------------------------- #

# >>> 06 Training Loops <<<
for epoch in range(options.epochs_number):
    for batch_index, (data, data_label) in enumerate(loader):
        batch_size = data.shape[0]
        real = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)
        Z = Variable(torch.FloatTensor(np.random(0,1,(batch_size, options.latent_dim)))).to(device)
        C_generated = Variable(torch.LongTensor(np.random.randint(0, options.n_classes, batch_size))).to(device)
        generated = generator(Z, C_generated)

        # >>> 06.01  Training Discriminator <<<
        # >>> 06.01.01 Data of Discriminator <<<
        Z = Variable(torch.FloatTensor(np.random(0,1,(batch_size, options.latent_dim)))).to(device)
        C_generated = Variable(torch.LongTensor(np.random.randint(0, options.n_classes, batch_size))).to(device)
        generated = generator(Z, C_generated)
        # >>> 06.01.02 Loss of Discriminator <<<
        # >>> 06.01.02.01 Loss of Real data <<<
        output_discriminator_real = discriminator(Variable(torch.FloatTensor(data)).to(device), data_label)
        loss_discriminator_real = optim_loss(output_discriminator_real, real)
        # >>> 06.01.02.02 Loss of Fake data <<<
        output_discriminator_fake = discriminator(generated.detach(), C_generated)
        loss_discriminator_fake = optim_loss(output_discriminator_fake, fake)
        loss_discriminator = (loss_discriminator_real + loss_discriminator_fake)/2
        # >>> 06.01.03 Gradient of Discriminator <<<
        optim_discriminator.zero_grad()
        loss_discriminator.backward()
        optim_discriminator.step()       

        # >>> 06.02  Training Generator <<<
        # >>> 06.02.01 Loss of Generator <<<
        output_discriminator = discriminator(generated, C_generated)
        loss_generator = optim_loss(output_discriminator, real)
        # >>> 06.02.02 Gradient of Generator <<<
        optim_generator.zero_grad()
        loss_generator.backward()
        optim_generator.step()

        # # >>> 06.03  Training Infomation <<<
        # # >>> 06.03.01  Infomation in Terminal <<<
        # if batch_index == 0:
        #     print(f"Epoch {epoch}, Loss of Discriminator {loss_discriminator:.2f}, Loss of Generator {loss_generator:.2f}")
        # # >>> 06.03.02  Infomation in Tensorboard <<<
        # with torch.no_grad():
        #     generated = generator(fixed_noise).reshape(-1,1,28,28)
        #     realdata = data.reshape(-1,1,28,28)
        #     img_generated = torchvision.utils.make_grid(generated, normalize=True)
        #     img_realdata = torchvision.utils.make_grid(realdata, normalize=True)
        #     writer_gene.add_image("Fake Images", img_generated, global_step= step) 
        #     writer_gene.add_image("Real Images", img_realdata, global_step= step)
        #     step += 1 



