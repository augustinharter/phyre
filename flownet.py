#%%
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from phyre_rolllout_collector import load_phyre_rollouts
from cv2 import resize, imshow, waitKey
import cv2
from phyre_utils import *
from itertools import chain
import argparse
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
                    X[:,:,:,pos+add] += T.tanh(self.conv(X[:,:,:,pos]))
                else:
                    X[:,:,pos+add,:] += T.tanh(self.conv(X[:,:,pos,:]))
                pos += add
            return X
        
        #Alternative concept
        else:
            if sideways:
                first = T.tanh(self.conv(X[:,:,:,pos]))
                Y = T.zeros(X.shape[0], first.shape[1], X.shape[2], first.shape[2])
                Y[:,:,:,pos] = first
            else:
                first = T.tanh(self.conv(X[:,:,pos,:]))
                Y = T.zeros(X.shape[0], first.shape[1], first.shape[2], X.shape[3])
                Y[:,:,pos,:] = first
            while pos!=end:
                if sideways:
                    Y[:,:,:,pos+add] += Y[:,:,:,pos] + T.tanh(self.conv(X[:,:,:,pos+add]))
                else:
                    Y[:,:,pos+add,:] += Y[:,:,pos,:] + T.tanh(self.conv(X[:,:,pos+add,:]))
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

class FlowNet(nn.Module):
    def __init__(self, in_dim, chs):
        super().__init__()
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
            nn.Conv2d(in_dim, chs, 3, 1, 1),
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

class UpFlowNet(nn.Module):
    def __init__(self, in_dim, chs):
        super().__init__()
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
            nn.Conv2d(in_dim, chs, 3, 1, 1),
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
            #X = self.d1(X)
            X = self.l1(X)
            X = self.r1(X)
            X = self.u2(X)
            #X = self.d2(X)
            X = self.l2(X)
            X = self.r2(X)
            #X = self.mid_conv(X)
        X = self.end_conv(X)
        #X = F.relu(X[:,None,0])
        return X

class ActionBallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.Conv2d(4, 1, 5, 1, 2),
            nn.Sigmoid()
        )
    def forward(self, X):
        return self.model(X)

class Discriminator(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(chan, 4, 4, 2, 1),
            nn.Conv2d(4, 8, 4, 2, 1),
            nn.Conv2d(8, 8, 4, 2, 1),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(4*16, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1))
    
    def forward(self, X):
        return self.model(X)

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

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_id', type=str)
    parser.add_argument('-t', action='store_true')
    parser.add_argument('-e', action='store_true')
    parser.add_argument('-i', action='store_true')
    parser.add_argument('-gan', action='store_true')
    parser.add_argument('-gt', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()
#%%
if __name__ == "__main__":
    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    if args.t:
        train_dataloader, index = make_mono_dataset(f"data/phyre_fold_{fold_id}_train", size=(32,32), tasks=train_ids)
#%%
if __name__ == "__main__":
    tar_net = FlowNet(5, 16)
    act_net = FlowNet(7, 16)
    ext_net = UpFlowNet(7, 16)
    discr = Discriminator(8)
#%%
if __name__ == "__main__" and args.t:
    #opti = T.optim.Adam(tar_net.parameters(recurse=True), lr=1e-3)
    #opti2 = T.optim.Adam(act_net.parameters(recurse=True), lr=1e-3)
    #opti3 = T.optim.Adam(ext_net.parameters(recurse=True), lr=1e-3)
    nets_opti = T.optim.Adam(chain(tar_net.parameters(recurse=True), 
                            act_net.parameters(recurse=True),
                            ext_net.parameters(recurse=True)), 
                        lr=3e-3)
    discr_opti = T.optim.Adam(discr.parameters(), lr=3e-3)
#%%
def train1():
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

def train3():
    for e in range(100):
        print("epoch", e)
        for i, (X, Y) in enumerate(dataloader):
            with T.no_grad():
                Z = model(X)
                A = model2(T.cat((X[:,1:], Y[:,None,2], Z), dim=1))
            #B = model3(T.cat((X[:,1:], Y[:,None,2], Z, A), dim=1))
            B = extract_action(A)
            #loss = F.binary_cross_entropy(B[:,0], Y[:,3])
            #opti3.zero_grad()
            #loss.backward()
            #opti3.step()
            #print(loss.item())
            if not i%10:
                clear_output(wait=True)
                orig = T.cat(tuple(T.cat((sub, T.ones(32,1)*0.5), dim=1) for sub in X[0]), dim=1)
                print(orig.shape)
                plt.imshow(orig)
                plt.show()
                print(X.shape, Y.shape, Z.shape)
                plt.imshow(T.cat((Y[0,0].detach(), T.ones(32,1), Z[0,0].detach(), T.ones(32,1), Y[0,2].detach(),
                    T.ones(32,1), Y[0,1].detach(),T.ones(32,1), A[0,0].detach(), T.ones(32,1), Y[0,3].detach(),
                    T.ones(32,1), B[0,0].detach()), dim=1))
                plt.show()

def train_supervised(tar_net:FlowNet, act_net:FlowNet, ext_net:UpFlowNet, 
    data_loader:T.utils.data.DataLoader, opti: T.optim.Adam, use_GT=True):
    for epoch in range(args.epochs):
        for i, (X,) in enumerate(data_loader):
            # Prepare Data
            init_scenes = X[:,1:6]
            target_paths = X[:,6]
            action_paths = X[:,8]
            goal_paths = X[:,7]
            base_paths = X[:,9]
            action_balls = X[:,0]
            #print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)

            # Forward Pass
            target_pred = tar_net(init_scenes)
            if use_GT:
                action_pred = act_net(T.cat((init_scenes, target_paths[:,None], base_paths[:,None]), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_paths[:,None], action_paths[:,None]), dim=1))
            else:
                action_pred = act_net(T.cat((init_scenes, target_pred, base_paths[:,None]), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
            
            if not i%50:
                plt.imsave(f'result/flownet/training/run_{args.path_id}_epoch_{epoch}_batchx100_{i}.png',np.concatenate(tuple(np.concatenate((sub, T.ones(32,1)*0.5), axis=1) 
                    for sub in T.cat((X, target_pred, action_pred, ball_pred), dim=1)[0].detach()), axis=1))
            #plt.show()

            # Loss
            tar_loss = F.binary_cross_entropy(target_pred, target_paths[:,None])
            act_loss = F.binary_cross_entropy(action_pred, action_paths[:,None])
            ball_loss = F.binary_cross_entropy(ball_pred, action_balls[:,None])
            loss = ball_loss + tar_loss + act_loss
            print(i, loss.item())

            # Backward Pass
            opti.zero_grad()
            loss.backward()
            opti.step()

def train_as_gan(tar_net:FlowNet, act_net:FlowNet, ext_net:UpFlowNet, discr:Discriminator,
    data_loader:T.utils.data.DataLoader, nets_opti: T.optim.Optimizer, discr_opti: T.optim.Optimizer, use_GT=True):
    switch = False
    for epoch in range(args.epochs):
        for i, (X,) in enumerate(data_loader):
            switch = not switch
            # Prepare Data
            init_scenes = X[:,1:6]
            target_paths = X[:,6]
            action_paths = X[:,8]
            goal_paths = X[:,7]
            base_paths = X[:,9]
            action_balls = X[:,0]
            #print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)

            # TRAIN GENERATORS
            # Forward Pass 
            if switch:
                target_pred = tar_net(init_scenes)
                action_pred = act_net(T.cat((init_scenes, target_paths[:,None], base_paths[:,None]), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_paths[:,None], action_paths[:,None]), dim=1))
                if not i%50:
                    plt.imsave(f'result/flownet/training/run_{args.path_id}_epoch_{epoch}_batchx100_{i}.png',np.concatenate(tuple(np.concatenate((sub, T.ones(32,1)*0.5), axis=1) 
                        for sub in T.cat((X, target_pred, action_pred, ball_pred), dim=1)[0].detach()), axis=1))
                #plt.show()

                # Loss
                validity = discr(T.cat((init_scenes, target_pred, action_pred, ball_pred), dim=1))
                gen_loss = F.binary_cross_entropy_with_logits(validity, T.ones(X.shape[0],1))

                # Backward Pass
                nets_opti.zero_grad()
                gen_loss.backward()
                nets_opti.step()
                print(i,"gen_loss:", gen_loss.item())

            # TRAIN DISCRIMINATOR
            else:
                target_pred = tar_net(init_scenes)
                action_pred = act_net(T.cat((init_scenes, target_paths[:,None], base_paths[:,None]), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_paths[:,None], action_paths[:,None]), dim=1))

                # Loss
                validity = discr(T.cat((init_scenes, target_pred, action_pred, ball_pred), dim=1))
                fake_loss = F.binary_cross_entropy_with_logits(validity, T.zeros(X.shape[0],1))
                validity = discr(T.cat((init_scenes, target_paths[:,None], action_paths[:,None], action_balls[:,None]), dim=1))
                real_loss = F.binary_cross_entropy_with_logits(validity, T.ones(X.shape[0],1))
                discr_loss = fake_loss+real_loss

                # Backward Pass
                discr_opti.zero_grad()
                discr_loss.backward()
                discr_opti.step()
                print(i, "discr_loss:", discr_loss.item()/2)
        
#%%
if __name__ == "__main__" and args.t:
    #train()
    #model.load_state_dict(T.load("saves/imaginet-c16-all.pt"))
    #train2()
    #model2.load_state_dict(T.load("saves/imaginet2-c16-all.pt"))
    if args.gan:
        train_as_gan(    tar_net, act_net, ext_net, discr, train_dataloader, nets_opti, discr_opti, use_GT=args.gt)
    else:
        train_supervised(tar_net, act_net, ext_net, train_dataloader, nets_opti, use_GT=args.gt)

# %%
if __name__ == "__main__" and args.t:
    T.save(tar_net.state_dict(), f"saves/flownet_tar_{args.path_id}.pt")
    T.save(act_net.state_dict(), f"saves/flownet_act_{args.path_id}.pt")
    T.save(ext_net.state_dict(), f"saves/flownet_ext_{args.path_id}.pt")
    if args.gan:
        T.save(discr.state_dict(), f"saves/flownet_discr_{args.path_id}.pt")

#%%
def inspect(tar_net:FlowNet, act_net:FlowNet, ext_net:UpFlowNet, 
    data_loader:T.utils.data.DataLoader):
    for i, (X,) in enumerate(data_loader):
        init_scenes = X[:,1:6]
        target_paths = X[:,6]
        action_paths = X[:,8]
        goal_paths = X[:,7]
        base_paths = X[:,9]
        action_balls = X[:,0]

        with T.no_grad():
            target_pred = tar_net(init_scenes)
            action_pred = act_net(T.cat((init_scenes, target_pred, base_paths[:,None]), dim=1))
            ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))

        for j in range(X.shape[0]):
            orig = T.cat(tuple(T.cat((sub, T.ones(32,1)*0.5), dim=1) for sub in X[j]), dim=1)
            #print(orig.shape)
            plt.imsave(f'result/flownet/testing/{i*X.shape[0]+j}.png',np.concatenate(tuple(np.concatenate((sub, T.ones(32,1)*0.5), axis=1) 
                    for sub in T.cat((X, target_pred, action_pred, ball_pred), dim=1)[j].detach()), axis=1))

if __name__ == "__main__" and args.i:
    fold_id=0
    tar_net.load_state_dict(T.load(f"saves/flownet_tar_{args.path_id}.pt"))
    act_net.load_state_dict(T.load(f"saves/flownet_act_{args.path_id}.pt"))
    ext_net.load_state_dict(T.load(f"saves/flownet_ext_{args.path_id}.pt"))
    test_dataloader, index = make_mono_dataset(f"data/phyre_fold_{fold_id}_test", size=(32,32), tasks=test_ids)
    inspect(tar_net, act_net, ext_net, test_dataloader)
# %%
