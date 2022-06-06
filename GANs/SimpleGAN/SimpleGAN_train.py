import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from SimpleGAN import Discriminator, Generator

# ---------------------------------------------- #
# ---------------------------------------------- #

# >>> 01 Iintailize Hyperparameters <<<
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
latent_dim = 64
generated_dim = 28*28
batch_size = 32
num_epochs = 100

# >>> 02 Iintailize Generator and Discriminator <<<
discriminator = Discriminator(generated_dim).to(device)
generator = Generator(latent_dim, generated_dim).to(device)

# >>> 03 Iintailize Dataset <<<
fixed_noise = torch.randn((batch_size, latent_dim)).to(device)
transforms = transforms.Compose([transforms.ToTensor, transforms.Normalize((0.1307,),(0.3081,))])
dataset = datasets.MNIST(root="./dataset/", train=False, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# >>> 04 Iintailize Optimizer <<<
optim_discriminator =  optim.Adam(discriminator.parameters(), lr = lr)
optim_generator = optim.Adam(generator.parameters(), lr = lr)
criterion = nn.BCELoss()

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
        # >>> 06.01.01  Data of Discriminator <<<
        discriminator_data = discriminator(data).view(-1)
        discriminator_gene = discriminator(generated).view(-1)
        # >>> 06.01.02 Loss of Discriminator <<<
        loss_discriminator_data = criterion(discriminator_data, torch.ones_like(discriminator_data))
        loss_discriminator_gene = criterion(discriminator_gene, torch.ones_like(discriminator_gene))
        loss_discriminator = (loss_discriminator_data + loss_discriminator_gene) / 2.0
        # >>> 06.01.03 Gradient of Discriminator <<<
        discriminator.zero_grad()
        loss_discriminator.backward(retain_graph = True)
        optim_discriminator.step()

        # >>> 06.02  Training Generator <<<
        # >>> 06.02.01  Data of Generator <<<
        output_discriminator = discriminator(generated).view(-1)
        # >>> 06.02.02 Loss of Generator <<<
        loss_generator = criterion(output_discriminator, torch.ones_like(output_discriminator))
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



