import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import os
import random
import time
import numpy as np

random.seed(42)


class Loader:
    def __init__(self, batch_size, ):
        self.files = os.listdir("utkcropped")
        random.shuffle(self.files)
        self.batch_size = batch_size
        self.batch_track = 0

    def reshuffle(self):
        random.shuffle(self.files)
        self.batch_track = 0

    def load_batch(self, inc_track=True):
        b = torch.zeros(self.batch_size, 3, 200, 200)
        for i, x in enumerate(self.files[self.batch_track:self.batch_track + self.batch_size]):
            b[i] = torch.Tensor(plt.imread("utkcropped/" + x) / 255).view(1, 3, 200, 200)
        self.batch_track += self.batch_size
        return b


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 200
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),  # 200
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # 100
            nn.Conv2d(32, 64, 3, 1, 1),  # 100
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),  # 100
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # 50
            nn.Conv2d(32, 32, 1, 1),  # 50
            nn.ReLU(),

        )
        self.fc1 = nn.Linear(32 * 50 * 50, 500, bias=False)
        self.fc2 = nn.Linear(32 * 50 * 50, 500, bias=False)
        self.fc3 = nn.Linear(500, 32 * 50 * 50, bias=False)

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 100
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 200
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),  # 200, out
        )

    def reparametrize(self, mu, sig):
        std = torch.exp(0.5 * sig)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x.to(torch.device("cuda")))
        mu, sig = self.fc1(x.view(-1, 32 * 50 * 50)), self.fc2(x.view(-1, 32 * 50 * 50))
        x = self.reparametrize(mu, sig)
        x = self.fc3(x)
        return self.decoder(x.view(-1, 32, 50, 50)), mu, sig


def loss(pred, x, mu, sig):
    bce = torch.sum((x.view(-1, 3 * 200 * 200) - pred.view(-1, 3 * 200 * 200)) ** 2)
    kld = -0.5 * torch.sum(1 + sig - mu.pow(2) - sig.exp())
    return kld * 50 + bce

torch.cuda.empty_cache()

model = VAE()
model.cuda()
gen = Loader(16)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10
t = time.time()

for e in range(epochs):
    for b in range(len(gen.files) // gen.batch_size):
        batch = gen.load_batch()
        opt.zero_grad()
        out, mu, sig = model.forward(batch)
        l = loss(out, batch.to(torch.device("cuda")), mu, sig)
        l.backward()
        opt.step()
        if b % 20 == 19:
            print(f"Epoch: {e + 1}\tBatch: {b + 1}\tLoss: {l.item()}\t in {time.time() - t}")
            t = time.time()
    gen.batch_track = 0
    gen.reshuffle()
