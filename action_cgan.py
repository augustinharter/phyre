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
    parser.add_argument('-init', action='store_true')
    parser.add_argument('-lindisc', action='store_true')
    parser.add_argument('-lingen', action='store_true')
    parser.add_argument('-sequ', action='store_true')
    parser.add_argument('-load', action='store_true')
    parser.add_argument('-viz', action='store_true')
    parser.add_argument('-cons', action='store_true')
    parser.add_argument('-overfit', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--width', default=16, type=int)
    parser.add_argument('--geneval', default=10, type=int)
    parser.add_argument('--saveevery', default=10, type=int)

    args = parser.parse_args()
    print(args)

    BASE_PATH = './saves/action_cgan/'
    SAVE_PATH = BASE_PATH+args.path
    DATA_PATH = f'./data/all_task_interactions/{args.data}/interactions.pickle'
    NOISE_DIM = 100
    ONLY_GENERATE = args.genonly
    WIDTH = args.width
    S_CHANNELS = 4
    A_CHANNELS = 2
    BATCH_SIZE = 64

    #X = np.zeros((128,S_CHANNELS+A_CHANNELS,WIDTH,WIDTH)) Dummy Dataset
    with open(DATA_PATH, 'rb') as fs:
        X = T.tensor(pickle.load(fs), dtype=T.float)
    print('loaded dataset with shape:', X.shape)
    if args.overfit:
        data_set = T.utils.data.TensorDataset(X.repeat(128,1,1,1))
        print('repeated dataset to shape:', X.repeat(128,1,1,1).shape)
    else:
        data_set = T.utils.data.TensorDataset(X)
    data_loader = T.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)

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
                nn.ConvTranspose2d(8, a_chan, 4, 2, 1),
                nn.Sigmoid()])
        if folds==2:
            gen_mods.extend([
                nn.ConvTranspose2d(16, 8, 4, 2, 1),
                nn.LeakyReLU(0.1),
                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8, a_chan, 4, 2, 1),
                nn.Sigmoid()])
        if folds==1:
            gen_mods.extend([
                nn.ConvTranspose2d(16, a_chan, 4, 2, 1),
                nn.Sigmoid()])
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
            nn.Sigmoid(),
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
        return out

#%%
def train(epoch, generators, g_optimizer, discriminators, d_optimizer, data_loader, criterion, args,
            verbose_every=10, gen_eval_every=10, device='cuda'):

    # extract models
    if args.sequ:
        generator, generator2 = generators
        discriminator, discriminator2 = discriminators
    else:
        generator = generators[0]
        discriminator = discriminators[0]

    # Start Epoch
    for i, (batch,) in enumerate(data_loader):
        batch.float()
        bh = batch.shape[0]//2
        if args.single:
            gen_batch = batch[:bh,[0,1,2,3,5]].to(device)
            disc_batch = batch[bh:2*bh,[0,1,2,3,5]].to(device)
        if args.init:
            gen_batch = batch[:bh,[0,1,2,3,6,5]].to(device)
            disc_batch = batch[bh:2*bh,[0,1,2,3,6,5]].to(device)
        else:
            gen_batch = batch[:bh,:6].to(device)
            disc_batch = batch[bh:2*bh,:6].to(device)
        T.autograd.set_detect_anomaly(True)

        #vis_batch(disc_batch, "result/test", "disc_batch")
        #vis_batch(gen_batch, "result/test", "gen_batch")

        # Discriminator
        # Forward
        d_optimizer.zero_grad()
        noise = T.randn(disc_batch.shape[0], generator.noise_dim).to(device)
        if args.sequ:
            # first stage
            fake = generator(disc_batch[:,:generator.s_chan], noise).detach()
            fake_validity = discriminator(T.cat((disc_batch[:,:generator.s_chan], fake), dim=1))
            real_validity = discriminator(disc_batch[:,[0,1,2,3,5]])
            disc_fake_loss = criterion(fake_validity, T.zeros_like(fake_validity))
            disc_real_loss = criterion(real_validity, T.ones_like(real_validity))
            # second stage
            noise2 = T.randn(disc_batch.shape[0], generator2.noise_dim).to(device)
            primed_cond = disc_batch[:,[0,1,2,3,5]]
            primed_fake = generator2(primed_cond, noise2).detach()
            fake_validity2 = discriminator2(T.cat((disc_batch[:,:4], primed_fake, disc_batch[:,None,5]), dim=1))
            real_validity2 = discriminator2(disc_batch)
            disc_fake_loss2 = criterion(fake_validity2, T.zeros_like(fake_validity2))
            disc_real_loss2 = criterion(real_validity2, T.ones_like(real_validity2))
            disc_comb_loss = disc_fake_loss + disc_real_loss + disc_fake_loss2 + disc_real_loss2
        else:
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
        noise = T.randn(gen_batch.shape[0], generator.noise_dim).to(device)
        gens = generator(gen_batch[:,:generator.s_chan], noise)
        gen_validity = discriminator(T.cat((gen_batch[:,:generator.s_chan], gens), dim=1))
        gen_loss = criterion(gen_validity, T.ones_like(gen_validity))
        if args.sequ:
            noise2 = T.randn(disc_batch.shape[0], generator.noise_dim).to(device)
            primed_cond = gen_batch[:,[0,1,2,3,5]]
            primed_fake = generator2(primed_cond, noise2)
            gen_validity2 = discriminator2(T.cat((gen_batch[:,:discriminator2.s_chan-1], primed_fake, gen_batch[:,None,discriminator2.s_chan-1]), dim=1))
            gen_loss2 = criterion(gen_validity2, T.ones_like(gen_validity2))
            gen_loss = gen_loss + gen_loss2
        # Backward
        gen_loss.backward()
        g_optimizer.step()

        if verbose_every and not i%verbose_every:
            if args.sequ:
                print(f'D: fl {disc_fake_loss} rl {disc_real_loss}  G: {gen_loss-gen_loss2}\n'+
                      f'D2: fl {disc_fake_loss2} rl {disc_real_loss2}  G2: {gen_loss2}')
            else:
                print(f'D: fl {disc_fake_loss} rl {disc_real_loss}  G: {gen_loss}')

        if gen_eval_every and not (epoch+1)%gen_eval_every and not i:
            generate(generator, gen_batch, 1, args.path+'-training', str(epoch)+'_2', sequ=generator2 if args.sequ else None, device=device, GT=False)
            generate(generator, gen_batch, 1, args.path+'-training', epoch, sequ=generator2 if args.sequ else None, device=device, GT=False)
            generate(generator, gen_batch, 1, args.path+'-training', str(epoch)+'_GT', sequ=generator2 if args.sequ else None, device=device)
            generate(generator, gen_batch, 1, args.path+'-training', str(epoch)+'_GT_2', sequ=generator2 if args.sequ else None, device=device)

def generate(generator, cond_batch, n_per_sample, path, save_id, grid=0, sequ = None, device="cpu", GT=True):
    # generate fakes
    with T.no_grad():
        noise = T.randn(cond_batch.shape[0], generator.noise_dim).to(device)
        fakes = generator(cond_batch[:,:generator.s_chan], noise).detach()
        if sequ is not None:
            noise2 = T.randn(cond_batch.shape[0], generator.noise_dim).to(device)
            if GT:
                primed_cond = cond_batch[:,[0,1,2,3,5]]
            else:
                primed_cond = T.cat((cond_batch[:,:generator.s_chan], fakes), dim=1)
            #primed_cond =cond_batch[:,:generator.s_chan+1]
            primed_fake = sequ(primed_cond, noise2)
            fakes = T.cat((fakes, primed_fake), dim=1)

    # visualize
    single = fakes.shape[1] == 1
    '''
    if not grid:
        gridsize = (int(fakes.shape[0]**0.5)+1)
        grid = (gridsize, gridsize)
    fig, ax = plt.subplots(grid[0],grid[1], sharex= True, sharey=True)
    for i,fake in enumerate(fakes):
        ax[i//grid[1], i%grid[1]].imshow(T.cat((
            cond_batch[i,0], T.ones(fake.shape[1],1),
            cond_batch[i,1], T.ones(fake.shape[1],1),
            cond_batch[i,2], T.ones(fake.shape[1],1),
            cond_batch[i,3], T.ones(fake.shape[1],1),
            cond_batch[i,4], T.ones(fake.shape[1],1),
            cond_batch[i,5] if not single else T.ones(fake.shape[1],1), T.ones(fake.shape[1],1),
            fake[0],         T.ones(fake.shape[1],1), 
            fake[1] if not single else T.ones(fake.shape[1],1)), dim=1))
    '''
    actions = cond_batch[:,-2:].cpu()
    g = fakes if not single else T.cat((fakes, fakes), axis=1)
    g = g.cpu()
    wid = fakes.shape[2]
    num_cells = fakes.shape[0]
    s = cond_batch.cpu()
    border = T.sum(s, dim=1)[:,None]==s.shape[1]
    border.reshape(num_cells,1,wid,wid).numpy()
    border = np.pad(np.concatenate((border, border, border), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0)

    def remove(pic, back):
        pic[back] = 0
        return pic

    # composing original scene    
    back = s[:,3].reshape(num_cells,1,wid,wid).numpy()
    #print(s[:,0].shape)
    green = np.max(np.stack((0.5*s[:,0],s[:,1],0.5*s[:,2],back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)

    green0 = s[:,0].reshape(num_cells,1,wid,wid).numpy()
    green1 = s[:,1].reshape(num_cells,1,wid,wid).numpy()
    green2 = s[:,2].reshape(num_cells,1,wid,wid).numpy()
    redminus = s[:,4].reshape(num_cells,1,wid,wid).numpy()
    redzero = s[:,5].reshape(num_cells,1,wid,wid).numpy()

    red = np.max(np.stack((0*actions[:,0],actions[:,1],back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
    red2 = np.max(np.stack((0*actions[:,1],actions[:,0],back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
    orig = np.pad(np.concatenate((red, green, red2), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=1)
    orig_empty = np.pad(np.concatenate((back[:,None,0], green, back[:,None,0]), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)

    gt_red = np.pad(np.concatenate((redminus, 0*redminus, 0*redminus), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)
    gt_red2 = np.pad(np.concatenate((0*back, 0*back, redzero), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)
    gt_green0 = np.pad(np.concatenate((0*back, green0, 0*back), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)
    gt_green1 = np.pad(np.concatenate((0*back, green1,0*back), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)
    gt_green2 = np.pad(np.concatenate((0*back, green2, 0*back), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)

    # composing scene with generated action
    red = np.max(np.stack((g[:,0],0*g[:,1],back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
    red2 = np.max(np.stack((0*g[:,0],g[:,1],back[:,0]), axis=-1), axis=-1).reshape(num_cells,1,wid,wid)
    gen = np.pad(np.concatenate((red, green, red2), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)
    gen0 = np.pad(np.concatenate((red, green, back), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)
    #print(combined)
    pred_red = np.pad(np.concatenate((redminus, 0*back, 0*back), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)
    pred_red2 = np.pad(np.concatenate((0*back, 0*back, redzero), axis=1), ((0,0), (0,0), (1,1), (1,1)), constant_values=0.5)
    combined = np.concatenate((orig, gen), axis=1).reshape(2*num_cells,3,wid+2,wid+2)
    if args.viz:
        combined = orig.reshape(num_cells,3,wid+2,wid+2)
    grid = make_grid(T.tensor(combined), nrow=8, normalize=True)
    #plt.imshow(grid[0])
    #plt.show(block=False)

    # save
    os.makedirs(f'./result/action_cgan/{path}', exist_ok=True)
    #fig.savefig(f'./result/action_cgan/{path}/{save_id}', dpi=1000)
    save_image(grid, f'./result/action_cgan/{path}/{save_id}grid.png')
    #plt.show()


    def transform(pic, border=None):
        print(pic.shape)
        if border is not None:
            pic = remove(pic, border)
        pic = T.from_numpy(pic)
        pic = pic.transpose(1,3).transpose(1,2)
        #pic = pic[None,0]
        return pic


    orig = transform(orig)
    orig_empty = transform(orig_empty)
    gt_green0 = transform(gt_green0, border)
    gt_green1 = transform(gt_green1, border)
    gt_green2 = transform(gt_green2, border)
    gt_red = transform(gt_red, border)
    gt_red2 = transform(gt_red2, border)
    pred_red = transform(pred_red, border)
    pred_red2 = transform(pred_red2, border)
    gen = transform(gen)
    gen0 = transform(gen0)

    print(orig.shape)
    text = ["complete\nscene", "condition\nscene without\nred ball", "scene with\nstage 1 result", "scene with\nstage 1 & 2\nresult"]
    pb = T.stack((orig,orig_empty, gen0, gen), dim=1)
    print(pb.shape)
    for i in range(s.shape[0]):
        lpb = pb[i,None]
        vis_batch(lpb, "result/visuals", f"compact-{i}-cgan-pipe-visual", text=text)

def save_models(models, save_path):
    os.makedirs(save_path, exist_ok=True)
    for model in models:
        print("saving:", save_path+f'/{model}.pt')
        T.save(models[model].state_dict(), save_path+f'/{model}.pt')

def load_models(models, load_path):
    for model in models:
        print("loading:", load_path+f'/{model}.pt')
        models[model].load_state_dict(T.load(load_path+f'/{model}.pt'))


# %%
if __name__ == "__main__":
    # Initializing models
    a_chans = A_CHANNELS-int(args.single)-int(args.sequ)
    generator = Generator(WIDTH, NOISE_DIM, S_CHANNELS, a_chans, conv= not args.lingen, folds=args.folds).to(device)
    discriminator = Discriminator(WIDTH, S_CHANNELS, a_chans, conv= not args.lindisc, folds=args.folds).to(device)
    if args.sequ:
        generator2 = Generator(WIDTH, NOISE_DIM, S_CHANNELS+1, a_chans, conv= not args.lingen, folds=args.folds).to(device)
        discriminator2 = Discriminator(WIDTH, S_CHANNELS+1, a_chans, conv= not args.lindisc, folds=args.folds).to(device)

    criterion = nn.BCELoss()
    disc_params = chain(discriminator.parameters(), discriminator2.parameters() if args.sequ else [])
    gen_params = chain(generator.parameters(), generator2.parameters() if args.sequ else [])
    d_optimizer = T.optim.Adam(disc_params, lr=1.5e-4)
    g_optimizer = T.optim.Adam(gen_params, lr=1.5e-4)

#%%
if __name__ == "__main__":
    if args.load:
        models = {'generator': generator, 'discriminator': discriminator}
        if args.sequ:
            models['generator2'] = generator2
            models['discriminator2'] = discriminator2
        load_models(models, SAVE_PATH)

    if args.load_disc!='':
        models = {'discriminator': discriminator}
        if args.sequ:
            models['discriminator2'] = discriminator2
        load_models(models, BASE_PATH+args.load_disc)
        
    # Training
    if not ONLY_GENERATE:
        for epoch in range(args.epochs):
            print(f'STARTING epoch {epoch+1}')
            if args.sequ:
                train(epoch, [generator, generator2], g_optimizer, [discriminator, discriminator2], d_optimizer, 
                    data_loader, criterion, args, verbose_every=args.verbose, gen_eval_every=args.geneval, device=device)
            else:
                train(epoch, [generator], g_optimizer, [discriminator], d_optimizer, 
                    data_loader, criterion, args, verbose_every=args.verbose, gen_eval_every=args.geneval)

            # models are saved every 'saveevery' epoch
            if args.saveevery and not (epoch+1)%args.saveevery:
                models = {'generator': generator, 'discriminator': discriminator}
                if args.sequ:
                    models['generator2'] = generator2
                    models['discriminator2'] = discriminator2
                save_models(models, SAVE_PATH)

    # Generating
    else:
        for i, (batch, ) in enumerate(data_loader):
            batch = batch.to(device)
            if args.init:
                batch = batch[:,[0,1,2,3,6,5]]
            else:
                batch = batch[:,:6]
            if i ==10:
                break
            size = 64 if args.viz else 32
            generate(generator, batch[:size], 1, args.path+'-eval', f"{i}_grid"+'_2', sequ=generator2 if args.sequ else None, device=device, GT=False)
            generate(generator, batch[:size], 1, args.path+'-eval', f"{i}_grid", sequ=generator2 if args.sequ else None, device=device, GT=False)
            generate(generator, batch[:size], 1, args.path+'-eval', f"{i}_grid"+'_GT', sequ=generator2 if args.sequ else None, device=device)
            generate(generator, batch[:size], 1, args.path+'-eval', f"{i}_grid"+'_GT_2', sequ=generator2 if args.sequ else None, device=device)
