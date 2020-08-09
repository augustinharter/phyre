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
import pickle
import os

#%%
if __name__ == "__main__":
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print("device:", device)
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='default',type=str)
    parser.add_argument('--verbose', default=10,type=int)
    parser.add_argument('-single', action='store_true')
    parser.add_argument('-lindisc', action='store_true')
    parser.add_argument('-lingen', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--geneval', default=10, type=int)
    parser.add_argument('--saveevery', default=10, type=int)
    args = parser.parse_args()
    print(args)

    BASE_PATH = './saves/action_cgan/'
    SAVE_PATH = BASE_PATH+args.path
    DATA_PATH = './data/fiddeling'
    NOISE_DIM = 100
    ONLY_GENERATE = False
    DATA_SIZE = 100 if ONLY_GENERATE else 5000
    WIDTH = 16
    S_CHANNELS = 4
    A_CHANNELS = 2
    BATCH_SIZE = 64

    X = np.zeros((128,S_CHANNELS+A_CHANNELS,WIDTH,WIDTH))
    with open(f'{DATA_PATH}/interactions.pickle', 'rb') as fs:
        X = T.tensor(pickle.load(fs), dtype=T.float)
    print('loaded dataset with shape:', X.shape)
    data_set = T.utils.data.TensorDataset(X)
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
            nn.BatchNorm1d(2048),
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
def train(epoch, generator, g_optimizer, discriminator, d_optimizer, data_loader, criterion, args,
            verbose_every=10, gen_eval_every=10):
    for i, (batch,) in enumerate(data_loader):
        batch.float()
        if args.single:
            gen_batch = batch[:batch.shape[0]//2,:-1]
            disc_batch = batch[batch.shape[0]//2:,:-1]
        else:
            gen_batch = batch[:batch.shape[0]//2]
            disc_batch = batch[batch.shape[0]//2:]
        T.autograd.set_detect_anomaly(True)

        # Discriminator
        # Forward
        d_optimizer.zero_grad()
        noise = T.randn(disc_batch.shape[0], generator.noise_dim)
        fakes = generator(disc_batch[:,:generator.s_chan], noise).detach()
        fake_validity = discriminator(T.cat((disc_batch[:,:generator.s_chan], fakes), dim=1))
        real_validity = discriminator(disc_batch)
        disc_fake_loss = criterion(fake_validity, T.zeros_like(fake_validity))
        disc_real_loss = criterion(real_validity, T.ones_like(real_validity))
        disc_comb_loss = disc_fake_loss + disc_real_loss
        # Backward
        disc_comb_loss.backward()
        d_optimizer.step()

        # Generator
        # Forward
        g_optimizer.zero_grad()
        noise = T.randn(disc_batch.shape[0], generator.noise_dim)
        gens = generator(gen_batch[:,:generator.s_chan], noise)
        gen_validity = discriminator(T.cat((gen_batch[:,:generator.s_chan], gens), dim=1))
        gen_loss = criterion(gen_validity, T.ones_like(gen_validity))
        # Backward
        gen_loss.backward()
        g_optimizer.step()

        if verbose_every and not i%verbose_every:
            print(f'D: fl {disc_fake_loss} rl {disc_real_loss}  G: {gen_loss}')

        if gen_eval_every and not epoch%gen_eval_every and not i:
            generate(generator, gen_batch, 1, args, epoch)
            generate(generator, gen_batch, 1, args, str(epoch)+'_')

def generate(generator, cond_batch, n_per_sample, args, save_id):
    noise = T.randn(cond_batch.shape[0], generator.noise_dim)
    fakes = generator(cond_batch[:,:generator.s_chan], noise).detach()
    grid = int(fakes.shape[0]**0.5)+1
    fig, ax = plt.subplots(grid,grid, sharex= True, sharey=True)
    for i,fake in enumerate(fakes):
        ax[i//grid, i%grid].imshow(T.cat((
            #cond_batch[i,0], T.ones(fake.shape[1],1),
            #cond_batch[i,1], T.ones(fake.shape[1],1),
            cond_batch[i,4], T.ones(fake.shape[1],1),
            cond_batch[i,5] if not args.single else T.ones(fake.shape[1],1), T.ones(fake.shape[1],1),
            fake[0],         T.ones(fake.shape[1],1), 
            fake[1] if not args.single else T.ones(fake.shape[1],1)), dim=1))
    os.makedirs(f'./result/action_cgan/{args.path}', exist_ok=True)
    fig.savefig(f'./result/action_cgan/{args.path}/{save_id}')
    #plt.show()

def save_models(generator, discriminator, save_path):
    os.makedirs(save_path, exist_ok=True)
    T.save(generator.state_dict(), save_path+'/generator.pt')
    T.save(discriminator.state_dict(), save_path+'/discriminator.pt')


# %%
if __name__ == "__main__":
    a_chans = A_CHANNELS-int(args.single)
    generator = Generator(WIDTH, NOISE_DIM, S_CHANNELS, a_chans, conv=args.lingen).to(device)
    discriminator = Discriminator(WIDTH, S_CHANNELS, a_chans, conv=args.lindisc).to(device)

    criterion = nn.BCELoss()
    d_optimizer = T.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = T.optim.Adam(generator.parameters(), lr=1e-4)

#%%
if __name__ == "__main__":
    for epoch in range(args.epochs):
        print(f'STARTING epoch {epoch}')
        train(epoch, generator, g_optimizer, discriminator, d_optimizer, data_loader, criterion, args, verbose_every=args.verbose,
            gen_eval_every=args.geneval)
        if args.saveevery and epoch and not epoch%args.saveevery:
            save_models(generator, discriminator, SAVE_PATH)
