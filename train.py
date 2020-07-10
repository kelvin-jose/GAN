import torch
import config
from engine import train
from model import Discriminator
from model import Generator
from dataset import Dataset

mnist = Dataset()
dataloader = mnist.get_loader()

device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')

generator = Generator(config.LATENT_SIZE, config.HIDDEN_SIZE, config.INPUT_SIZE)
discriminator = Discriminator(config.INPUT_SIZE, config.HIDDEN_SIZE)

generator = generator.to(device)
discriminator = discriminator.to(device)

g_optim = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

loss_func = torch.nn.BCELoss()

generator.train()
discriminator.train()
train(dataloader, discriminator, generator, d_optim, g_optim, device, loss_func)

torch.save(generator.state_dict(), 'generator.ckpt')
torch.save(discriminator.state_dict(), 'discriminator.ckpt')

