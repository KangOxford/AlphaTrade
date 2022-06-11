import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from WGAN import Discriminator, Generator
from Grandient_Penalty import gradient_penalty

# ------------------------------------------- #
#                                             #
# WGAN-GP May.2022                            #
# Written by Kang                             #
# Funtion:                                    #
#   For the training process of the WGAN-GP   #
#                                             #
# ------------------------------------------- #

# >>> 01 Iintailize Hyperparameters <<<
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
latent_dim = 128
figure_dim = 64
noise_dim = 100
noise_Dim = 32
channels_dim = 1
generated_dim = 64
discriminator_dim = 64
discriminator_iterations = 10
batch_size = 128
num_epochs = 10
lambda_ = 10

# >>> 02 Iintailize Generator and Discriminator <<<
def initializing_weights(model):
    for item in model.modules():
        if isinstance(item, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(item.weight.data, 0.0, 0.01)
discriminator = Discriminator(channels_dim, discriminator_dim).to(device)
generator = Generator(latent_dim, channels_dim, generated_dim).to(device)
initializing_weights(discriminator)
initializing_weights(generator)

# >>> 03 Iintailize Dataset <<<
fixed_noise = torch.randn(noise_Dim, noise_dim, 1, 1).to(device)
transforms = transforms.Compose([
    transforms.Resize(figure_dim),
    transforms.ToTensor,
    transforms.Normalize([0.1307 for _ in range(channels_dim)],[0.3081 for _ in range(channels_dim)])
    ])
dataset = datasets.MNIST(root="./dataset/", train=False, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# >>> 04 Iintailize Optimizer <<<
optim_discriminator =  optim.RMSprop(discriminator.parameters(), lr = lr)
optim_generator = optim.RMSprop(generator.parameters(), lr = lr)

# >>> 05 Set Tensorboard <<<
writer_gene = SummaryWriter(f"logs/gene")
writer_data = SummaryWriter(f"logs/data")
step = 0 

# ---------------------------------------------- #
# ---------------------------------------------- #

# >>> 06 Training Loops <<<
for epoch in range(num_epochs):
    for batch_index, (data, data_lable) in enumerate(loader):
        data = data.view(-1, 784).to(device)
        batch_size = data.shape[0]
        generated = generator(torch.randn(batch_size, latent_dim).to(device))

        # >>> 06.01  Training Discriminator <<<
        for _ in range(discriminator_iterations):
            # >>> 06.01.01  Data of Discriminator <<<
            noise_data = torch.rand(batch_size, latent_dim, 1, 1).to(device)
            discriminator_data = discriminator(data).view(-1)
            discriminator_gene = discriminator(generated).view(-1)
            # >>> 06.01.02 Gradient Penalty of Discriminator <<<
            gradientPenalty = gradient_penalty(discriminator, discriminator_data, discriminator_gene)
            # >>> 06.01.03 Loss of Discriminator <<<
            loss_discriminator = -1 * (torch.mean(discriminator_data) - torch.mean(discriminator_gene)) + lambda_ * gradientPenalty 
            # >>> 06.01.04 Gradient of Discriminator <<<
            discriminator.zero_grad()
            loss_discriminator.backward(retain_graph = True)
            optim_discriminator.step()
            # >>> 06.01.05 weights clipping of Discriminator <<<
            for item in discriminator.parameters():
                item.data.clamp_(-0.01, 0.01)

        # >>> 06.02  Training Generator <<<
        # >>> 06.02.01  Data of Generator <<<
        output_discriminator = discriminator(generated).view(-1)
        # >>> 06.02.02 Loss of Generator <<<
        loss_generator = -1 * torch.mean(output_discriminator)
        # >>> 06.02.03 Gradient of Generator <<<
        generator.zero_grad()
        loss_generator.backward()
        optim_generator.step()

        # >>> 06.03  Training Infomation <<<
        # >>> 06.03.01  Infomation in Terminal <<<
        if batch_index == 0:
            print(f"Epoch {epoch}, Loss of Discriminator {loss_discriminator:.2f}, Loss of Generator {loss_generator:.2f}")
        # >>> 06.03.02  Infomation in Tensorboard <<<
        with torch.no_grad():
            generated = generator(fixed_noise).reshape(-1,1,28,28)
            realdata = data.reshape(-1,1,28,28)
            img_generated = torchvision.utils.make_grid(generated, normalize=True)
            img_realdata = torchvision.utils.make_grid(realdata, normalize=True)
            writer_gene.add_image("Fake Images", img_generated, global_step= step) 
            writer_gene.add_image("Real Images", img_realdata, global_step= step)
            step += 1 