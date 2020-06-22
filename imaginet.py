#%%
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import MNIST
import torchvision as TV
from matplotlib import pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
from sampler import load_phyre_rollouts
from cv2 import resize, imshow, waitKey
import cv2
#%%
class SpatialConv(nn.Module):
    def __init__(self, conv, direction, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.conv = conv
        self.direction = direction

    def forward(self, X):
        
        sideways = False
        direction = self.direction
        if direction=="right":
            sideways = True
            pos = 0
            end = X.shape[3]-1
            add = 1
        elif direction=="left":
            sideways = True
            end = 0
            pos = X.shape[3]-1
            add = -1
        elif direction=="up":
            end = 0
            pos = X.shape[2]-1
            add = -1
        elif direction=="down":
            pos = 0
            end = X.shape[2]-1
            add = 1
        else:
            print("Direction not understood!")
            return X

        #ORIGINAL "INPLACE" CONCEPT from paper
        if self.inplace:
            X = X.clone()
            while pos!=end:
                if sideways:
                    X[:,:,:,pos+add] += F.tanh(self.conv(X[:,:,:,pos]))
                else:
                    X[:,:,pos+add,:] += F.tanh(self.conv(X[:,:,pos,:]))
                pos += add
            return X
        
        #Alternative concept
        else:
            if sideways:
                first = F.tanh(self.conv(X[:,:,:,pos]))
                Y = T.zeros(X.shape[0], first.shape[1], X.shape[2], first.shape[2])
                Y[:,:,:,pos] = first
            else:
                first = F.tanh(self.conv(X[:,:,pos,:]))
                Y = T.zeros(X.shape[0], first.shape[1], first.shape[2], X.shape[3])
                Y[:,:,pos,:] = first
            while pos!=end:
                if sideways:
                    Y[:,:,:,pos+add] += Y[:,:,:,pos] + F.tanh(self.conv(X[:,:,:,pos+add]))
                else:
                    Y[:,:,pos+add,:] += Y[:,:,pos,:] + F.tanh(self.conv(X[:,:,pos+add,:]))
                pos += add
            return Y

class ImagiNet(nn.Module):
    def __init__(self):
        super().__init__()
        chs = 16
        #self.feature_dims = feature_dims
        #self.embed_dims = embed_dims
        #self.final_conv_width = ((int(self.feature_dims**0.5)//4)-2)
        conv_fn = lambda: nn.Conv1d(chs, chs, 5, 1, 2)
        self.r1 = SpatialConv(conv_fn(), "right", inplace=True)
        self.l1 =  SpatialConv(conv_fn(), "left", inplace=True)
        self.u1 =    SpatialConv(conv_fn(), "up", inplace=True)
        self.d1 =  SpatialConv(conv_fn(), "down", inplace=True)
        self.r2 = SpatialConv(conv_fn(), "right", inplace=True)
        self.l2 =  SpatialConv(conv_fn(), "left", inplace=True)
        self.u2 =    SpatialConv(conv_fn(), "up", inplace=True)
        self.d2 =  SpatialConv(conv_fn(), "down", inplace=True)
        self.init_conv = nn.Sequential(
            nn.Conv2d(7, chs, 3, 1, 1),
            nn.Tanh())
        self.end_conv = nn.Sequential(
            nn.Conv2d(chs, 1, 3, 1, 1),
            nn.Sigmoid())
        self.mid_conv = nn.Sequential(
            nn.Conv2d(7, 7, 3, 1, 1),
            nn.Tanh())
        self.big_conv = nn.Sequential(
            nn.Conv2d( 1, 16, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16,  8, 4, 2, 1),
            nn.LeakyReLU(0.1))
        self.deconv = nn.Sequential(
            #nn.Linear(embed_dims, 8*self.final_conv_width**2),
            nn.Tanh(),
            #View((-1, 8, self.final_conv_width, self.final_conv_width)),
            nn.ConvTranspose2d(8, 16, 3, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16,  1, 4, 2, 1)) 

    def forward(self, X):
        X = self.init_conv(X)
        for n in range(1):
            X = self.u1(X)
            X = self.d1(X)
            X = self.l1(X)
            X = self.r1(X)
            #X = self.u2(X)
            #X = self.d2(X)
            #X = self.l2(X)
            #X = self.r2(X)
            #X = self.mid_conv(X)
        X = self.end_conv(X)
        #X = F.relu(X[:,None,0])
        return X

class ImagiNet2(nn.Module):
    def __init__(self):
        super().__init__()
        chs = 16
        #self.feature_dims = feature_dims
        #self.embed_dims = embed_dims
        #self.final_conv_width = ((int(self.feature_dims**0.5)//4)-2)
        conv_fn = lambda: nn.Conv1d(chs, chs, 5, 1, 2)
        self.r1 = SpatialConv(conv_fn(), "right", inplace=True)
        self.l1 =  SpatialConv(conv_fn(), "left", inplace=True)
        self.u1 =    SpatialConv(conv_fn(), "up", inplace=True)
        self.d1 =  SpatialConv(conv_fn(), "down", inplace=True)
        self.r2 = SpatialConv(conv_fn(), "right", inplace=True)
        self.l2 =  SpatialConv(conv_fn(), "left", inplace=True)
        self.u2 =    SpatialConv(conv_fn(), "up", inplace=True)
        self.d2 =  SpatialConv(conv_fn(), "down", inplace=True)
        self.init_conv = nn.Sequential(
            nn.Conv2d(8, chs, 3, 1, 1),
            nn.Tanh())
        self.end_conv = nn.Sequential(
            nn.Conv2d(chs, 1, 3, 1, 1),
            nn.Sigmoid())
        self.mid_conv = nn.Sequential(
            nn.Conv2d(7, 7, 3, 1, 1),
            nn.Tanh())
        self.big_conv = nn.Sequential(
            nn.Conv2d( 1, 16, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16,  8, 4, 2, 1),
            nn.LeakyReLU(0.1))
        self.deconv = nn.Sequential(
            #nn.Linear(embed_dims, 8*self.final_conv_width**2),
            nn.Tanh(),
            #View((-1, 8, self.final_conv_width, self.final_conv_width)),
            nn.ConvTranspose2d(8, 16, 3, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16,  1, 4, 2, 1)) 

    def forward(self, X):
        X = self.init_conv(X)
        for n in range(1):
            X = self.u1(X)
            X = self.d1(X)
            X = self.l1(X)
            X = self.r1(X)
            #X = self.u2(X)
            #X = self.d2(X)
            #X = self.l2(X)
            #X = self.r2(X)
            #X = self.mid_conv(X)
        X = self.end_conv(X)
        #X = F.relu(X[:,None,0])
        return X

def neighbs(pos, shape, back, value):
    y, x = pos
    return [(k,l,value) for k in range(y-1, y+2) for l in range (x-1, x+2) if l>=0 and k>=0 and l<shape[1] and k<shape[0] and not ((k,l) in back)]

def calc_cost_map(map):
    front = []
    back = set()
    cost = T.zeros_like(map)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i,j]:
                cost[i,j] = 0
                back.add((i,j))
                front.extend(neighbs((i,j), map.shape, back, 0))
    while front:
        break

def dist_map(pic):
    #Calc Goal Pos
    X, Y = 0, 0
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            if pic[y,x]:
                X += pic[y,x]*x
                Y += pic[y,x]*y
    summed = np.sum(pic)
    if summed == 0:
        return T.zeros(pic.shape[0], pic.shape[1])
    X /= summed
    Y /= summed

    #Calc Dist to Goal
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            pic[y,x] = ((y-Y)**2 + (x-X)**2)**0.5
    return pic/(1.41*pic.shape[0])

def prepare_data(data, size):
    targetchannel = 1
    X, Y = [], []
    print("Preparing dataset...")
    #x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for variations in data:
        with_base = len(variations) > 1
        for (j, rollout) in enumerate(variations):
            if not isinstance(rollout, np.ndarray):
                break
            #length = (2*len(rollout))//3
            #rollout = rollout[:length]
            roll = np.zeros((len(rollout), 7, size[0], size[1]))
            for i, scene in enumerate(rollout):
                channels = [(scene==j).astype(float) for j in range(1,8)]
                roll[i] = np.stack([(resize(c, size, cv2.INTER_MAX)>0).astype(float) for c in channels])
            roll = np.flip(roll, axis=2)
            trajectory = (np.sum(roll[:,targetchannel], axis=0)>0).astype(float)
            if not(with_base and j == 0):
                action = (np.sum(roll[:,0], axis=0)>0).astype(float)
            #goal_prior = dist_map(roll[0, 2] + roll[0, 3])
            #roll[0, 0] = goal_prior
            # TESTING ONLY
            #roll[0, 1] = roll[0, 0]
            if with_base and j == 0:
                base = trajectory
            else:
                roll[0, 0] = np.zeros_like(roll[0,0])
                #print(goal_prior)
                # Contains the initial scene without action
                X.append(roll[0])
                # Contains goaltarget, actiontarget, basetrajectory
                Y.append(np.stack((trajectory, action, base if with_base else np.zeros_like(roll[0,0]))))
                #plt.imshow(trajectory)
                #plt.show()
    print("Finished preparing!")
    return X, Y

def load_dataset(path, size=(32,32)):
    X = load_phyre_rollouts(path)
    X, Y = prepare_data(X, size)
    X = T.tensor(X).float()
    Y = T.tensor(Y).float()
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X,Y), 32, shuffle=True)
    return dataloader

#%%
dataloader = load_dataset("data/phyre_all_obs", size=(32,32))
#%%
model = ImagiNet()
model2 = ImagiNet2()
#%%
opti = T.optim.Adam(model.parameters(recurse=True), lr=1e-3)
opti2 = T.optim.Adam(model2.parameters(recurse=True), lr=1e-3)
#%%
def train():
    for e in range(100):
        print("epoch", e)
        for i, (X, Y) in enumerate(dataloader):
            Z = model(X)[:,0]
            #print(Z)
            loss = F.binary_cross_entropy(Z, Y[:,0])
            opti.zero_grad()
            loss.backward()
            opti.step()
            print(loss.item())
            if not i%20:
                clear_output(wait=True)
                orig = T.cat(tuple(T.cat((sub, T.ones(32,1)*0.5), dim=1) for sub in X[0]), dim=1)
                print(orig.shape)
                plt.imshow(orig)
                plt.show()
                print(X.shape, Y.shape, Z.shape)
                plt.imshow(T.cat((X[0,0], T.ones(32,1), Y[0,0].detach(), T.ones(32,1), Z[0].detach()), dim=1))
                plt.show()

def train2():
    for e in range(100):
        print("epoch", e)
        for i, (X, Y) in enumerate(dataloader):
            with T.no_grad():
                Z = model(X)
            #print(Z)
            A = model2(T.cat((X[:,1:], Y[:,None,2], Z), dim=1))
            loss = F.binary_cross_entropy(A[:,0], Y[:,1])
            opti2.zero_grad()
            loss.backward()
            opti2.step()
            print(loss.item())
            if not i%20:
                clear_output(wait=True)
                orig = T.cat(tuple(T.cat((sub, T.ones(32,1)*0.5), dim=1) for sub in X[0]), dim=1)
                print(orig.shape)
                plt.imshow(orig)
                plt.show()
                print(X.shape, Y.shape, Z.shape)
                plt.imshow(T.cat((Y[0,0].detach(), T.ones(32,1), Z[0,0].detach(), T.ones(32,1), Y[0,2].detach(),
                    T.ones(32,1), Y[0,1].detach(),T.ones(32,1), A[0,0].detach()), dim=1))
                plt.show()

#%%
#train()
model.load_state_dict(T.load("saves/imaginet-c16-all.pt"))
train2()
# %%
#T.save(model.state_dict(), "saves/imaginet-c16-all-v2.pt")
T.save(model2.state_dict(), "saves/imaginet2-c16-all.pt")
# %%
model.load_state_dict(T.load("saves/imaginet-c16-all.pt"))
dataloader = load_dataset("data/phyre_grid_obs", size=(32,32))
#%%
def test():
    for i, (X, Y) in enumerate(dataloader):
        if i==2:
            break
        #print(Z)
        with T.no_grad():
            Z = model(X)
            A = model2(T.cat((X[:,1:], Y[:,None,2], Z), dim=1))
        #print(loss.item())
        #clear_output(wait=True)
        for j in range(X.shape[0]):
            orig = T.cat(tuple(T.cat((sub, T.ones(32,1)*0.5), dim=1) for sub in X[j]), dim=1)
            #print(orig.shape)
            plt.imsave(f"result/scnn/a-path/{j}_.png",orig.numpy())
            #print(X.shape, Y.shape, Z.shape)
            plt.imsave(f"result/scnn/a-path/{j}.png",T.cat((Y[j,0].detach(), T.ones(32,1), Z[j,0].detach(), T.ones(32,1), Y[j,2].detach(),
                    T.ones(32,1), Y[j,1].detach(),T.ones(32,1), A[j,0].detach()), dim=1).numpy())
test()
# %%
