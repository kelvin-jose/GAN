import config
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.l_relu_1 = nn.LeakyReLU(0.2)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.l_relu_2 = nn.LeakyReLU(0.2)
        self.linear_3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.linear_1(input)
        x = self.l_relu_1(x)
        x = self.linear_2(x)
        x = self.l_relu_2(x)
        x = self.linear_3(x)
        return self.sigmoid(x)


class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(latent_size, hidden_size)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, input):
        x = self.linear_1(input)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.linear_3(x)
        return self.tanh(x)
