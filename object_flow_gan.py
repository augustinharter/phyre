#%%
import matplotlib.pyplot as plt
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from scene_extractor import Extractor

import random
import argparse
import cv2
import pickle
import os
from itertools import chain
from phyre_utils import vis_batch

#%%
if __name__ == "__main__":
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print("device:", device)
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='default',type=str)
    parser.add_argument('--data', default='',type=str)
    parser.add_argument('--verbose', default=10,type=int)
    parser.add_argument('--folds', default=0,type=int)
    parser.add_argument('--load_disc', default='',type=str)
    parser.add_argument('-single', action='store_true')
    parser.add_argument('-genonly', action='store_true')
    parser.add_argument('-lindisc', action='store_true')
    parser.add_argument('-lingen', action='store_true')
    parser.add_argument('-sequ', action='store_true')
    parser.add_argument('-load', action='store_true')
    parser.add_argument('-delta', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--width', default=16, type=int)
    parser.add_argument('--geneval', default=10, type=int)
    parser.add_argument('--saveevery', default=10, type=int)

    args = parser.parse_args()
    print(args)

    BASE_PATH = './saves/object_flow/'
    SAVE_PATH = BASE_PATH+args.path
    DATA_PATH = f'./data/all_task_interactions/{args.data}/interactions.pickle'
    INFO_PATH = f'./data/all_task_interactions/{args.data}/info.pickle'
    NOISE_DIM = 100
    ONLY_GENERATE = args.genonly
    WIDTH = args.width
    IN_CH_N = 4
    OUT_CH_N = 3
    BATCH_SIZE = 64
    NUM_CELLS = 9

    #X = np.zeros((128,IN_CH_N+OUT_CH_N,WIDTH,WIDTH)) Dummy Dataset
    with open(DATA_PATH, 'rb') as fs:
        X = T.tensor(pickle.load(fs), dtype=T.float)
    if args.delta:
        with open(INFO_PATH, 'rb') as fs:
            info = T.tensor(pickle.load(fs)['deltas'], dtype=T.float)
    print('loaded dataset with shape:', X.shape)
    if X.shape[0]==1:
        data_set = T.utils.data.TensorDataset(X.repeat(128,1,1,1))
        if args.delta:
            info_set =T.utils.data.TensorDataset(info.repeat(128,1))
        print('loaded dataset with shape:', X.repeat(128,1,1,1).shape)
    else:
        data_set = T.utils.data.TensorDataset(X)
        if args.delta:
            info_set =T.utils.data.TensorDataset(info)
    data_loader = T.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)
    if args.delta:
        info_loader = T.utils.data.DataLoader(info_set, batch_size=BATCH_SIZE//2, shuffle=True)

# %%
class Discriminator(nn.Module):
    def __init__(self, width, s_chan, a_chan, conv=True, folds=0):
        super().__init__()
        self.width = width
        self.s_chan = s_chan
        self.a_chan = a_chan
        self.conv = conv
        if not folds:
            folds = int(np.log(width/8)/np.log(8))
        self.enc_width = int(width/(2**folds))
        
        self.reason = nn.Sequential(
            nn.Linear(16*self.enc_width**2, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

        enc_mods = []
        if folds == 1:
            enc_mods.extend([nn.Conv2d(s_chan+a_chan, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        if folds == 2:
            enc_mods.extend([nn.Conv2d(s_chan+a_chan, 8, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(8, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        if folds == 3:
            enc_mods.extend([nn.Conv2d(s_chan+a_chan, 8, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(8, 16, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(16, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        enc_mods.extend([nn.BatchNorm2d(16)])

        self.encoder = nn.Sequential(*enc_mods)

        self.lin_model = nn.Sequential(
            nn.Linear(self.width**2*(self.s_chan+self.a_chan), 1024),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
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

class DeltaDiscriminator(nn.Module):
    def __init__(self, width, s_chan, a_chan, n_cells, conv=True, folds=0):
        super().__init__()
        self.width = width
        self.s_chan = s_chan
        self.a_chan = a_chan
        self.conv = conv
        if not folds:
            folds = int(np.log(width/8)/np.log(8))
        self.enc_width = int(width/(2**folds))
        
        self.reason = nn.Sequential(
            nn.Linear(n_cells+16*self.enc_width**2, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

        enc_mods = []
        if folds == 1:
            enc_mods.extend([nn.Conv2d(s_chan+a_chan, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        if folds == 2:
            enc_mods.extend([nn.Conv2d(s_chan+a_chan, 8, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(8, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        if folds == 3:
            enc_mods.extend([nn.Conv2d(s_chan+a_chan, 8, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(8, 16, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(16, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        enc_mods.extend([nn.BatchNorm2d(16)])

        self.encoder = nn.Sequential(*enc_mods)

        self.lin_model = nn.Sequential(
            nn.Linear(self.width**2*(self.s_chan+self.a_chan), 1024),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, scenes: T.Tensor, cells):
        if self.conv:
            encodings = self.encoder(scenes)
            x = self.reason(T.cat((encodings.flatten(start_dim=1), cells), dim =1))
        else:
            x = self.lin_model(scenes.flatten(start_dim=1))
        return x

class DeltaGenerator(nn.Module):
    def __init__(self, width, noise_dim, s_chan, a_chan, cells, conv=True, folds=0):
        super().__init__()
        self.width = width
        self.s_chan = s_chan
        self.a_chan = a_chan
        self.noise_dim = noise_dim
        self.conv = conv
        if not folds:
            folds = int(np.log(width/8)/np.log(8))
        self.enc_width = int(width/(2**folds))
        
        self.reason = nn.Sequential(
            nn.Linear(noise_dim+16*self.enc_width**2, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,cells),
            nn.Softmax()
        )

        enc_mods = []
        if folds == 1:
            enc_mods.extend([nn.Conv2d(s_chan+a_chan, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        if folds == 2:
            enc_mods.extend([nn.Conv2d(s_chan+a_chan, 8, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(8, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        if folds == 3:
            enc_mods.extend([nn.Conv2d(s_chan+a_chan, 8, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(8, 16, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(16, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        enc_mods.extend([nn.BatchNorm2d(16)])

        self.encoder = nn.Sequential(*enc_mods)

        self.lin_model = nn.Sequential(
            nn.Linear(self.width**2*(self.s_chan+self.a_chan), 1024),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, scenes: T.Tensor, noise):
        if self.conv:
            encodings = self.encoder(scenes)
            x = self.reason(T.cat((encodings.flatten(start_dim=1), noise), dim=1))
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
    def __init__(self, width, noise_dim, s_chan, a_chan, conv=True, folds=0):
        super().__init__()
        self.width = width
        self.noise_dim = noise_dim
        self.s_chan = s_chan
        self.a_chan = a_chan
        self.conv = conv
        if not folds:
            folds = int(np.log(width/8)/np.log(8))
        self.enc_width = int(width/(2**folds))

        gen_mods = [
            nn.Linear(noise_dim+16*self.enc_width**2, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, self.enc_width**2*16),
            nn.ReLU(),
            nn.BatchNorm1d(self.enc_width**2*16),
            View((-1, 16, self.enc_width, self.enc_width))]
        if folds==3:
            gen_mods.extend([
                nn.ConvTranspose2d(16, 8, 4, 2, 1),
                nn.LeakyReLU(0.1),
                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8, 8, 4, 2, 1),
                nn.LeakyReLU(0.1),
                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8, a_chan, 4, 2, 1)])
        if folds==2:
            gen_mods.extend([
                nn.ConvTranspose2d(16, 8, 4, 2, 1),
                nn.LeakyReLU(0.1),
                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8, a_chan, 4, 2, 1)])
        if folds==1:
            gen_mods.extend([
                nn.ConvTranspose2d(16, a_chan, 4, 2, 1)])
        self.conv_model = nn.Sequential(*gen_mods)

        enc_mods = []
        if folds==1:
            enc_mods.extend([nn.Conv2d(s_chan, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        if folds==2:
            enc_mods.extend([nn.Conv2d(s_chan, 8, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(8, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        if folds==3:
            enc_mods.extend([nn.Conv2d(s_chan, 8, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(8, 16, 4, 2, 1), nn.LeakyReLU(0.1),
                            nn.Conv2d(16, 16, 4, 2, 1), nn.LeakyReLU(0.1)])
        enc_mods.extend([nn.BatchNorm2d(16)])

        self.encoder = nn.Sequential(*enc_mods)

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
            nn.Tanh(),
            View((-1, a_chan, width, width))
        )

    def forward(self, scenes, noise):
        if self.conv:
            scene_features = self.encoder(scenes)
            flat_features = scene_features.view(scene_features.shape[0], -1)
        else:
            flat_features = scenes.flatten(start_dim=1)
        x = T.cat([flat_features, noise], 1)
        out = self.conv_model(x) if self.conv else self.lin_model(x)
        #out = T.stack((T.sigmoid(out[:,0]), T.sigmoid(out[:,1]), -1*T.sigmoid(out[:,2])), dim=1)
        if self.s_chan==1:
            out = T.sigmoid(out)
        else:
            out = T.tanh(out)
        return out

#%%
def train(epoch, generators, g_optimizer, discriminators, d_optimizer, data_loader, criterion, args,
            verbose_every=10, gen_eval_every=10, device='cuda'):

    # extract models
    if args.sequ:
        generator, generator2 = generators
        discriminator, discriminator2 = discriminators
    elif args.delta:
        generator, generator2 = generators
        discriminator, discriminator2 = discriminators
    else:
        generator = generators[0]
        discriminator = discriminators[0]

    # Start Epoch
    for i, ((batch,), (info,)) in enumerate(zip(data_loader, info_loader if args.delta else data_loader)):
        if args.delta:
            threshold = 1
            negx, negy, posx, posy = info[:,0]<-threshold, info[:,1]<-threshold, info[:,0]>threshold, info[:,1]>threshold
            zerox = negx.logical_or(posx).logical_not()
            zeroy = negy.logical_or(posy).logical_not()
            options = [negx.logical_and(negy), negx.logical_and(zeroy), negx.logical_and(posy),
                    zerox.logical_and(negy), zerox.logical_and(zeroy), zerox.logical_and(posy),
                    posx.logical_and(negy), posx.logical_and(zeroy), posx.logical_and(posy)]
            grid = T.stack(options, dim=1).float()
            grid += 0.1*T.randn_like(grid)
            grid[grid>1] = 1
            grid[grid<0] = 0
        #print(grid)

        batch.float()
        bh = batch.shape[0]//2
        gen_batch = batch[:bh].to(device)
        disc_batch = batch[bh:2*bh].to(device)
    
        T.autograd.set_detect_anomaly(True)

        #vis_batch(disc_batch, "result/test", "disc_batch")
        #vis_batch(gen_batch, "result/test", "gen_batch")

        # Discriminator
        # Forward
        d_optimizer.zero_grad()
        noise = T.randn(disc_batch.shape[0], generator.noise_dim).to(device)
        if args.sequ:
            # first stage
            fakes = generator(disc_batch[:,:4], noise).detach()
            #print(fakes.shape)
            fake_validity = discriminator(T.cat((disc_batch[:,:4], fakes), dim=1))
            real_validity = discriminator(disc_batch[:,:5])
            disc_fake_loss = criterion(fake_validity, T.zeros_like(fake_validity))
            disc_real_loss = criterion(real_validity, T.ones_like(real_validity))
            disc_comb_loss = disc_fake_loss + disc_real_loss
            # second stage
            noise2 = T.randn(disc_batch.shape[0], generator2.noise_dim).to(device)
            primed_cond = disc_batch[:,:5]
            primed_fake = generator2(primed_cond, noise2).detach()
            fake_validity2 = discriminator2(T.cat((disc_batch[:,:5], primed_fake), dim=1))
            real_validity2 = discriminator2(disc_batch)
            disc_fake_loss2 = criterion(fake_validity2, T.zeros_like(fake_validity2))
            disc_real_loss2 = criterion(real_validity2, T.ones_like(real_validity2))
            disc_comb_loss = disc_fake_loss + disc_real_loss + disc_fake_loss2 + disc_real_loss2
        elif args.delta:
            # first stage
            fakes = generator(disc_batch[:,:4], noise).detach()
            #print(fakes.shape)
            fake_validity = discriminator(T.cat((disc_batch[:,:4], fakes), dim=1))
            real_validity = discriminator(disc_batch[:,:5])
            disc_fake_loss = criterion(fake_validity, T.zeros_like(fake_validity))
            disc_real_loss = criterion(real_validity, T.ones_like(real_validity))
            disc_comb_loss = disc_fake_loss + disc_real_loss
            # second stage
            noise2 = T.randn(disc_batch.shape[0], generator2.noise_dim).to(device)
            primed_cond = disc_batch[:,:5]
            faked_grid = generator2(primed_cond, noise2).detach()
            fake_validity2 = discriminator2(disc_batch, faked_grid)
            real_validity2 = discriminator2(disc_batch, grid)
            disc_fake_loss2 = criterion(fake_validity2, T.zeros_like(fake_validity2))
            disc_real_loss2 = criterion(real_validity2, T.ones_like(real_validity2))
            disc_comb_loss = disc_fake_loss + disc_real_loss + disc_fake_loss2 + disc_real_loss2
        else:
            fakes = generator(disc_batch[:,:4], noise).detach()
            #print(fakes.shape)
            fake_validity = discriminator(T.cat((disc_batch[:,:4], fakes), dim=1))
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
        noise = T.randn(gen_batch.shape[0], generator.noise_dim).to(device)
        gens = generator(gen_batch[:,:4], noise)
        #print(gens.shape)
        gen_validity = discriminator(T.cat((gen_batch[:,:4], gens), dim=1))
        gen_loss = criterion(gen_validity, T.ones_like(gen_validity))
        if args.sequ:
            noise2 = T.randn(disc_batch.shape[0], generator.noise_dim).to(device)
            primed_cond = gen_batch[:,:5]
            primed_fake = generator2(primed_cond, noise2)
            gen_validity2 = discriminator2(T.cat((gen_batch[:,:5], primed_fake), dim=1))
            gen_loss2 = criterion(gen_validity2, T.ones_like(gen_validity2))
            gen_loss = gen_loss + gen_loss2
        elif args.delta:
            noise2 = T.randn(disc_batch.shape[0], generator.noise_dim).to(device)
            primed_cond = gen_batch[:,:5]
            faked_grid = generator2(primed_cond, noise2)
            gen_validity2 = discriminator2(gen_batch[:,:5], faked_grid)
            gen_loss2 = criterion(gen_validity2, T.ones_like(gen_validity2))
            gen_loss = gen_loss + gen_loss2

        # Backward
        gen_loss.backward()
        g_optimizer.step()

        if verbose_every and not i%verbose_every:
            if args.sequ or args.delta:
                print(f'D: fl {disc_fake_loss} rl {disc_real_loss}  G: {gen_loss-gen_loss2}\n'+
                      f'D2: fl {disc_fake_loss2} rl {disc_real_loss2}  G2: {gen_loss2}')
                if args.delta:
                    print("similar percentage:", (grid.round()==faked_grid.round()).all(dim=1).float().sum()/grid.shape[0])
            else:
                print(f'D: fl {disc_fake_loss} rl {disc_real_loss}  G: {gen_loss}')

        if gen_eval_every and not (epoch+1)%gen_eval_every and not i:
            generate(generator, gen_batch, 1, args.path+'-training', epoch, sequ=generator2 if args.sequ else None, device=device)
            generate(generator, gen_batch, 1, args.path+'-training', str(epoch)+'_', sequ=generator2 if args.sequ else None, device=device)

def generate(generator, cond_batch, n_per_sample, path, save_id, grid=0, sequ = None, device="cpu", GT=True, delta=False):
    # generate fakes
    with T.no_grad():
        noise = T.randn(cond_batch.shape[0], generator.noise_dim).to(device)
        fakes = generator(cond_batch[:,:4], noise).detach()
        if sequ is not None:
            noise2 = T.randn(cond_batch.shape[0], generator.noise_dim).to(device)
            if GT:
                primed_cond = cond_batch[:,:5]
            else:
                primed_cond = T.cat((cond_batch[:,:4], fakes), dim=1)
            #primed_cond =cond_batch[:,:generator.s_chan+1]
            primed_fake = sequ(primed_cond, noise2)
            fakes = T.cat((fakes, primed_fake), dim=1)

    # visualize
    #single = fakes.shape[1] == 1
    '''
    if not grid:
        gridsize = (int(fakes.shape[0]**0.5)+1)
        grid = (gridsize, gridsize)
    fig, ax = plt.subplots(grid[0],grid[1], sharex= True, sharey=True)
    for i,fake in enumerate(fakes):
        ax[i//grid[1], i%grid[1]].imshow(T.cat((
            cond_batch[i,0], T.ones(fake.shape[1],1),
            cond_batch[i,1], T.ones(fake.shape[1],1),# composing scene with faked position
    red = np.max(np.stack((g[:,0],back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
    pos = np.stack((red, green, back), axis=-1)
            cond_batch[i,2], T.ones(fake.shape[1],1),
            cond_batch[i,3], T.ones(fake.shape[1],1),
            cond_batch[i,4], T.ones(fake.shape[1],1),
            cond_batch[i,5] if not single else T.ones(fake.shape[1],1), T.ones(fake.shape[1],1),
            fake[0],         T.ones(fake.shape[1],1), 
            fake[1] if not single else T.ones(fake.shape[1],1)), dim=1))
    '''
    g = fakes.cpu()
    GT = cond_batch[:,4:].cpu()
    #g = g.cpu()
    wid = fakes.shape[2]
    num_cells = fakes.shape[0]
    s = cond_batch.cpu()


    # composing original scene position
    back = s[:,3].reshape(num_cells,1,wid,wid).numpy()
    green = np.max(np.stack((0.5*s[:,0],s[:,1],0.5*s[:,2],back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
    red = np.max(np.stack((GT[:,0],back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
    pos_GT = np.stack((red, green, back), axis=-1)

    # composing scene with faked position
    red = np.max(np.stack((g[:,0],back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
    pos = np.stack((red, green, back), axis=-1)

    if not args.delta:
        # composing original scene xvel
        positives = GT[:,1]*(GT[:,1]>0)
        negatives = -1*GT[:,1]*(GT[:,1]<0)
        red = np.max(np.stack((negatives, back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
        blue = np.max(np.stack((positives, back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
        xvel_GT = np.stack((red, green, blue), axis=-1)

        # composing original scene yvel
        positives = GT[:,2]*(GT[:,2]>0)
        negatives = -1*GT[:,2]*(GT[:,2]<0)
        red = np.max(np.stack((negatives, back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
        blue = np.max(np.stack((positives, back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
        yvel_GT = np.stack((red, green, blue), axis=-1)


        # composing scene with faked xvel
        positives = g[:,1]*(g[:,1]>0)
        negatives = -1*g[:,1]*(g[:,1]<0)
        red = np.max(np.stack((negatives, back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
        blue = np.max(np.stack((positives, back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
        xvel = np.stack((red, green, blue), axis=-1)

        # composing scene with faked yvel
        positives = g[:,2]*(g[:,2]>0)
        negatives = -1*g[:,2]*(g[:,2]<0)
        red = np.max(np.stack((negatives, back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
        blue = np.max(np.stack((positives, back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
        yvel = np.stack((red, green, blue), axis=-1)

        #print(combined)
        combined = np.stack((pos_GT, pos, xvel_GT, xvel, yvel_GT, yvel), axis=1)[:,:,0]
        text = ["pos GT", "pos faked", "xvel GT", "xvel faked", "yvel GT", "yvel faked"]
        #print(combined.shape)
        combined[combined>1] = 1
        vis_batch(T.from_numpy(combined), f'./result/object_flow/{path}', f'{save_id}grid', text=text)
    else:
        combined = np.stack((pos_GT, pos), axis=1)[:,:,0]
        text = ["pos GT", "pos faked"]
        #print(combined.shape)
        combined[combined>1] = 1
        vis_batch(T.from_numpy(combined), f'./result/object_flow/{path}', f'{save_id}grid', text=text)

    #grid = make_grid(T.tensor(combined), nrow=8, normalize=True)
    #plt.imshow(grid[0])
    #plt.show(block=False)

    # save
    #os.makedirs(f'./result/object_flow/{path}', exist_ok=True)
    #fig.savefig(f'./result/object_flow/{path}/{save_id}', dpi=1000)
    #save_image(grid, f'./result/object_flow/{path}/{save_id}grid.png')
    #plt.show()

def save_models(models, save_path):
    os.makedirs(save_path, exist_ok=True)
    for model in models:
        print("saving:", save_path+f'/{model}.pt')
        T.save(models[model].state_dict(), save_path+f'/{model}.pt')

def load_models(models, load_path):
    os.makedirs(load_path, exist_ok=True)
    for model in models:
        print("loading:", load_path+f'/{model}.pt')
        models[model].load_state_dict(T.load(load_path+f'/{model}.pt'))


# %%
if __name__ == "__main__":
    # Initializing models
    out_ch_n = OUT_CH_N - (2 if args.sequ or args.delta else 0)
    generator = Generator(WIDTH, NOISE_DIM, IN_CH_N, out_ch_n, conv= not args.lingen, folds=args.folds).to(device)
    discriminator = Discriminator(WIDTH, IN_CH_N, out_ch_n, conv= not args.lindisc, folds=args.folds).to(device)
    if args.sequ:
        generator2 = Generator(WIDTH, NOISE_DIM, IN_CH_N+1, OUT_CH_N-1, conv= not args.lingen, folds=args.folds).to(device)
        discriminator2 = Discriminator(WIDTH, IN_CH_N+1, OUT_CH_N-1, conv= not args.lindisc, folds=args.folds).to(device)
    elif args.delta:
        generator2 = DeltaGenerator(WIDTH, NOISE_DIM, IN_CH_N, out_ch_n, NUM_CELLS, conv= not args.lingen, folds=args.folds).to(device)
        discriminator2 = DeltaDiscriminator(WIDTH, IN_CH_N, out_ch_n, NUM_CELLS, conv= not args.lindisc, folds=args.folds).to(device)



    criterion = nn.BCELoss()
    disc_params = chain(discriminator.parameters(), discriminator2.parameters() if args.sequ or args.delta else [])
    gen_params = chain(generator.parameters(), generator2.parameters() if args.sequ or args.delta else [])
    d_optimizer = T.optim.Adam(disc_params, lr=1.5e-4)
    g_optimizer = T.optim.Adam(gen_params, lr=1.5e-4)

#%%
if __name__ == "__main__":
    if args.load:
        models = {'generator': generator, 'discriminator': discriminator}
        if args.sequ or args.delta:
            models['generator2'] = generator2
            models['discriminator2'] = discriminator2
        load_models(models, SAVE_PATH)

    if args.load_disc!='':
        models = {'discriminator': discriminator}
        if args.sequ or args.delta:
            models['discriminator2'] = discriminator2
        load_models(models, BASE_PATH+args.load_disc)
        
    # Training
    if not ONLY_GENERATE:
        for epoch in range(args.epochs):
            print(f'STARTING epoch {epoch+1}')
            if args.sequ or args.delta:
                train(epoch, [generator, generator2], g_optimizer, [discriminator, discriminator2], d_optimizer, 
                    data_loader, criterion, args, verbose_every=args.verbose, gen_eval_every=args.geneval, device=device)
            else:
                train(epoch, [generator], g_optimizer, [discriminator], d_optimizer, 
                    data_loader, criterion, args, verbose_every=args.verbose, gen_eval_every=args.geneval,  device=device)

            # models are saved every 'saveevery' epoch
            if args.saveevery and not (epoch+1)%args.saveevery:
                models = {'generator': generator, 'discriminator': discriminator}
                if args.sequ or args.delta:
                    models['generator2'] = generator2
                    models['discriminator2'] = discriminator2
                save_models(models, SAVE_PATH)

    # Generating
    else:
        for i, (batch, ) in enumerate(data_loader):
            batch = batch.to(device)
            if i ==10:
                break
            generate(generator, batch[:32], 1, f'{args.path}-results', i, grid=(4,2), 
                sequ = generator2 if args.sequ else None, device=device, delta= args.delta)
            generate(generator, batch[:32], 1, f'{args.path}-results', str(i)+'_', grid=(4,2), 
                sequ = generator2 if args.sequ else None, device=device, delta= args.delta)
