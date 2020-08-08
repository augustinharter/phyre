#%%
import matplotlib.pyplot as plt
import numpy as np

import torch as T
import torch.nn as nn
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import tqdm
from scene_extractor import Extractor
import random
import argparse
import cv2

#%%
if __name__ == "__main__":
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print("device:", device)
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='default',type=str)
    parser.add_argument('-gen', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()

    BASE_PATH = './saves/action_cgan/'
    SAVE_PATH = BASE_PATH+args.path
    NOISE_DIM = 100
    ONLY_GENERATE = args.gen
    DATA_SIZE = 100 if ONLY_GENERATE else 5000
    WIDTH = 16
    S_CHANNELS = 4
    A_CHANNELS = 2
    BATCH_SIZE = 64

    X = np.zeros((128,S_CHANNELS+A_CHANNELS,WIDTH,WIDTH))
    data_set = T.utils.data.TensorDataset(T.tensor(X, dtype=T.float))
    data_loader = T.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=False)

# %%
class Discriminator(nn.Module):
    def __init__(self, width, s_chan, a_chan, conv=True):
        super().__init__()
        self.width = width
        self.s_chan = s_chan
        self.a_chan = a_chan
        self.conv = conv
        
        self.reason = nn.Sequential(
            nn.Linear(64*(width//4)**2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(s_chan+a_chan, 32, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )

        self.lin_model = nn.Sequential(
            nn.Linear(self.width**2*(self.s_chan+self.a_chan), 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, scenes: T.Tensor):
        if self.conv:
            encodings = self.encoder(scenes)
            x = self.reason(encodings.flatten(start_dim=1))
        else:
            x = self.lin_model(scenes.flatten(start_dim=1))
        return x


# %%
class View(nn.Module):
    #Changing the Shape
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

#%%
class Generator(nn.Module):
    def __init__(self, width, noise_dim, s_chan, a_chan, conv=True):
        super().__init__()
        self.width = width
        self.noise_dim = noise_dim
        self.s_chan = s_chan
        self.a_chan = a_chan
        self.conv = conv
                
        self.conv_model = nn.Sequential(
            nn.Linear(width**2*s_chan + noise_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 8*8*16),
            nn.ReLU(),
            nn.BatchNorm1d(8*8*16),
            View((-1, 16, 8, 8)),
            nn.ConvTranspose2d(16, a_chan, 4, 2, 1),
            #nn.BatchNorm2d(8),
            #nn.ConvTranspose2d(8, a_chan, 4, 2, 1),
            nn.Sigmoid()
        )

        self.lin_model = nn.Sequential(
            #1
            nn.Linear(width**2*s_chan + noise_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            #2
            nn.Linear(2048, 8*8*16),
            nn.ReLU(),
            nn.BatchNorm1d(8*8*16),
            #3
            nn.Linear(8*8*16, width**2 *a_chan),
            nn.Sigmoid(),
            View((-1, a_chan, width, width))
        )

    def forward(self, scenes, noise):
        flat_scene = scenes.view(scenes.shape[0], -1)
        x = T.cat([flat_scene, noise], 1)
        out = self.conv_model(x) if self.conv else self.lin_model(x)
        return out

#%%
def train(generator, g_optimizer, discriminator, d_optimizer, data_loader, criterion, verbose=10):
    for i, (batch,) in enumerate(data_loader):
        batch.float()
        gen_batch = batch[:batch.shape[0]//2]
        disc_batch = batch[batch.shape[0]//2:]
        T.autograd.set_detect_anomaly(True)
        

        # Discriminator
        # Forward
        noise = T.rand(disc_batch.shape[0], generator.noise_dim)
        fakes = generator(disc_batch[:,:generator.s_chan], noise).detach()
        fake_validity = discriminator(T.cat((disc_batch[:,:generator.s_chan], fakes), dim=1))
        real_validity = discriminator(disc_batch)
        disc_fake_loss = criterion(fake_validity, T.zeros_like(fake_validity))
        disc_real_loss = criterion(real_validity, T.ones_like(real_validity))
        disc_comb_loss = disc_fake_loss + disc_real_loss
        # Backward
        d_optimizer.zero_grad()
        disc_comb_loss.backward()
        d_optimizer.step()

        # Generator
        # Forward
        noise = T.rand(disc_batch.shape[0], generator.noise_dim)
        gens = generator(gen_batch[:,:generator.s_chan], noise)
        gen_validity = discriminator(T.cat((gen_batch[:,:generator.s_chan], gens), dim=1))
        gen_loss = criterion(gen_validity, T.ones_like(gen_validity))
        # Backward
        g_optimizer.zero_grad()
        gen_loss.backward()
        g_optimizer.step()

        if verbose and not i%verbose:
            print(f'D: fl {disc_fake_loss} rl {disc_real_loss}  G: {gen_loss}')



# %%
if __name__ == "__main__":
    generator = Generator(WIDTH, NOISE_DIM, S_CHANNELS, A_CHANNELS).to(device)
    discriminator = Discriminator(WIDTH, S_CHANNELS, A_CHANNELS).to(device)

    criterion = nn.BCELoss()
    d_optimizer = T.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = T.optim.Adam(generator.parameters(), lr=1e-4)

#%%
if __name__ == "__main__":
    for epoch in range(args.epochs):
        print(f'STARTING epoch {epoch}')
        train(generator, g_optimizer, discriminator, d_optimizer, data_loader, criterion)
