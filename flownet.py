#%%
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from phyre_rolllout_collector import load_phyre_rollouts, collect_solving_dataset
from cv2 import resize, imshow, waitKey
import cv2
from phyre_utils import draw_ball, grow_action_vector, vis_batch, make_mono_dataset, action_delta_generator, gifify
from itertools import chain
import argparse
import os
import random
import phyre
#%%
class SpatialConv(nn.Module):
    def __init__(self, conv, direction, inplace=True, trans=False):
        super().__init__()
        self.inplace = inplace
        self.conv = conv
        self.direction = direction
        self.trans = trans

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
            end = X.shape[3]-1 if self.trans else 0
            pos = 0 if self.trans else X.shape[3]-1
            add = -1
        elif direction=="up":
            end = X.shape[2]-1 if self.trans else 0
            pos = 0 if self.trans else X.shape[2]-1
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
            
            if self.trans:
                if add<0:
                    X = X.flip(-1).clone() if sideways else X.flip(-2).clone()
                else:
                    X = X.clone()

                # transpose if going sideways
                X = X.transpose(-2,-1) if sideways else X

                # stepping through the tensor
                while pos!=end:
                    #print(X[:,:,pos,:].shape, X[:,:,pos-kd:pos+1,:].shape, self.conv(X[:,:,pos-kd:pos+1,:]).shape)
                    X[:,:,pos+1,:] += T.tanh(self.conv(X[:,:,pos,:]))
                    pos += 1

                # re-flip if direction was 'negative'
                if add<0:
                    X = X.flip(-1) if sideways else X.flip(-2)

                # re-transpose if sideways
                return X.transpose(-2,-1) if sideways else X

            else:
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

class SequentialConv(nn.Module):
    def __init__(self, conv, direction, inplace=True, trans=False):
        super().__init__()
        self.inplace = inplace
        self.conv = conv
        self.direction = direction

    def forward(self, X):
        #print('input shape', X.shape)
        #print('conv weight shape', self.conv.weight.shape)
        sideways = False
        direction = self.direction
        if direction=="right":
            sideways = True
            end = X.shape[3]
            pos = 0
            add = 1
        elif direction=="left":
            sideways = True
            end = -1
            pos = X.shape[3]-1
            add = -1
        elif direction=="up":
            end = -1
            pos = X.shape[2]-1
            add = -1
        elif direction=="down":
            end = X.shape[2]
            pos = 0
            add = 1
        else:
            print("Direction not understood!")
            return X

        #ORIGINAL "INPLACE" CONCEPT from paper
        if self.inplace:
            kd = self.conv.kernel_size[1]-1 if sideways else self.conv.kernel_size[0]-1
            #print(self.conv.kernel_size)
            pos += kd*add
            X = X.clone()
            while pos!=end:
                #print(pos)
                if sideways:
                    if add<0:
                        #print(X[:,:,:,pos].shape, X[:,:,:,pos:pos+kd+1].shape, self.conv(X[:,:,:,pos:pos+kd+1]).shape)
                        X[:,:,:,pos] += T.tanh(self.conv(X[:,:,:,pos:pos+kd+1])[:,:,0])
                    else:
                        #print(X[:,:,:,pos].shape, X[:,:,:,pos-kd:pos+1].shape, self.conv(X[:,:,:,pos-kd:pos+1]).shape)
                        X[:,:,:,pos] += T.tanh(self.conv(X[:,:,:,pos-kd:pos+1])[:,:,0])

                else:
                    if add<0:
                        #print(X[:,:,pos,:].shape, X[:,:,pos:pos+kd+1,:].shape, self.conv(X[:,:,pos:pos+kd+1,:]).shape)
                        X[:,:,pos,:] += T.tanh(self.conv(X[:,:,pos:pos+kd+1,:])[:,:,0])
                    else:
                        #print(X[:,:,pos,:].shape, X[:,:,pos-kd:pos+1,:].shape, self.conv(X[:,:,pos-kd:pos+1,:]).shape)
                        X[:,:,pos,:] += T.tanh(self.conv(X[:,:,pos-kd:pos+1,:])[:,:,0])

                pos += add
            return X
        
        #Alternative concept
        else:
            kw = (self.conv.kernel_size[1] if sideways else self.conv.kernel_size[0])-1
            pos += kw * add
            if sideways:
                first = T.tanh(self.conv(X[:,:,:,pos+(kw*add):pos]))
                Y = T.zeros(X.shape[0], first.shape[1], X.shape[2], first.shape[2])
                Y[:,:,:,pos] = first
            else:
                first = T.tanh(self.conv(X[:,:,pos+(kw*add):pos,:]))
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
    def __init__(self, in_dim, chs, sequ=False, trans=False):
        super().__init__()
        #self.feature_dims = feature_dims
        #self.embed_dims = embed_dims
        #self.final_conv_width = ((int(self.feature_dims**0.5)//4)-2)
        vertical_conv_fn = (lambda: nn.Conv2d(chs, chs, (2,5), 1, (0, 2))) if sequ else (lambda: nn.Conv1d(chs, chs, 5, 1, 2))
        horizontal_conv_fn = (lambda: nn.Conv2d(chs, chs, (5,2), 1, (2, 0))) if sequ else (lambda: nn.Conv1d(chs, chs, 5, 1, 2))
        flow_model = SequentialConv if sequ else SpatialConv
        self.r1 = flow_model(horizontal_conv_fn(), "right", inplace=True, trans=trans)
        self.l1 =  flow_model(horizontal_conv_fn(), "left", inplace=True, trans=trans)
        self.u1 =    flow_model(vertical_conv_fn(), "up", inplace=True, trans=trans)
        self.d1 =  flow_model(vertical_conv_fn(), "down", inplace=True, trans=trans)
        self.r2 = flow_model(horizontal_conv_fn(), "right", inplace=True, trans=trans)
        self.l2 =  flow_model(horizontal_conv_fn(), "left", inplace=True, trans=trans)
        self.u2 =    flow_model(vertical_conv_fn(), "up", inplace=True, trans=trans)
        self.d2 =  flow_model(vertical_conv_fn(), "down", inplace=True, trans=trans)
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
    def __init__(self, in_dim, chs, sequ=False, trans=False):
        super().__init__()
        #self.feature_dims = feature_dims
        #self.embed_dims = embed_dims
        #self.final_conv_width = ((int(self.feature_dims**0.5)//4)-2)
        vertical_conv_fn = (lambda: nn.Conv2d(chs, chs, (2,5), 1, (0, 2))) if sequ else (lambda: nn.Conv1d(chs, chs, 5, 1, 2))
        horizontal_conv_fn = (lambda: nn.Conv2d(chs, chs, (5,2), 1, (2, 0))) if sequ else (lambda: nn.Conv1d(chs, chs, 5, 1, 2))
        flow_model = SequentialConv if sequ else SpatialConv
        self.r1 = flow_model(horizontal_conv_fn(), "right", inplace=True, trans=trans)
        self.l1 =  flow_model(horizontal_conv_fn(), "left", inplace=True, trans=trans)
        self.u1 =    flow_model(vertical_conv_fn(), "up", inplace=True, trans=trans)
        self.d1 =  flow_model(vertical_conv_fn(), "down", inplace=True, trans=trans)
        self.r2 = flow_model(horizontal_conv_fn(), "right", inplace=True, trans=trans)
        self.l2 =  flow_model(horizontal_conv_fn(), "left", inplace=True, trans=trans)
        self.u2 =    flow_model(vertical_conv_fn(), "up", inplace=True, trans=trans)
        self.d2 =  flow_model(vertical_conv_fn(), "down", inplace=True, trans=trans)
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
    def __init__(self, in_dim):
        super().__init__()
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, 4, 4, 2, 1),
            nn.Conv2d(4, 8, 4, 2, 1),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(4*4*16, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1))
        """
        wid = 32
        folds = range(1, int(np.math.log2(wid)))
        acti = nn.ReLU
        convs = [nn.Conv2d(2**(2+i), 2**(3+i), 4, 2, 1) for i in folds]
        encoder = [nn.Conv2d(in_dim, 8, 4, 2, 1), acti()] + [acti() if i%2 else convs[i//2] for i in range(2*len(folds))]
        reason = [nn.Flatten(), nn.Linear(2**(3+max(folds)), 64), nn.Tanh(), nn.Linear(64,1), nn.Sigmoid()]
        modules = encoder+reason
        self.model = nn.Sequential(*modules)
    
    def forward(self, X):
        return self.model(X)

class Pyramid(nn.Module):
    def __init__(self, in_dim, chs):
        super().__init__()
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, 8, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, chs, 4, 2, 1),
            nn.Sigmoid()
        )
        """
        
        wid = 32
        folds = range(1, int(np.math.log2(wid)))
        acti = nn.ReLU
        convs = [nn.Conv2d(2**(2+i), 2**(3+i), 4, 2, 1) for i in folds]
        encoder = [nn.Conv2d(in_dim, 8, 4, 2, 1), acti()] + [acti() if i%2 else convs[i//2] for i in range(2*len(folds))]
        trans_convs = [nn.ConvTranspose2d(2**(3+i), 2**(2+i), 4, 2, 1) for i in reversed(folds)]
        decoder = [acti() if i%2 else trans_convs[i//2] for i in range(2*len(folds))] + [nn.ConvTranspose2d(8, chs, 4, 2, 1), nn.Sigmoid()]
        modules = encoder+decoder
        self.model = nn.Sequential(*modules)
        #print(self.model.state_dict().keys())

        """
        convs = [(2**(2+i), 2**(3+i)) for i in folds]
        trans_convs = [(2**(3+i), 2**(2+i)) for i in reversed(folds)]
        print(convs)
        print(trans_convs)
        encoder = [(in_dim,8), 'acti'] + [f"acti" if i%2 else convs[i//2] for i in range(2*len(folds))]
        print(encoder)
        decoder = [f"acti" if i%2 else trans_convs[i//2] for i in range(2*len(folds))] + [(8,chs), 'Sigmoid']
        print(decoder)
        print(*(encoder+decoder), sep='\n')
        """

    def forward(self, X):
        return self.model(X)

class SmartPyramid(nn.Module):
    def __init__(self, in_dim, chs):
        super().__init__()
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, 8, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, chs, 4, 2, 1),
            nn.Sigmoid()
        )
        """
        
        wid = 32
        folds = range(1, int(np.math.log2(wid)))
        acti = nn.ReLU
        convs = [nn.Conv2d(2**(2+i), 2**(3+i), 4, 2, 1) for i in folds]
        encoder = [nn.Conv2d(in_dim, 8, 4, 2, 1), acti()] + [acti() if i%2 else convs[i//2] for i in range(2*len(folds))]
        trans_convs = [nn.ConvTranspose2d(2**(3+i), 2**(2+i), 4, 2, 1) for i in reversed(folds)]
        decoder = [acti() if i%2 else trans_convs[i//2] for i in range(2*len(folds))] + [nn.ConvTranspose2d(8, chs, 4, 2, 1), nn.Sigmoid()]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        encoding_dim = 2**(3+max(folds))
        self.reason = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.Tanh(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Tanh()
        )


    def forward(self, X):
        X = self.encoder(X)
        shape = X.shape
        X = self.reason(X.view(shape[0],shape[1]))
        X = self.decoder(X.view(*shape))
        return X

class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        acti = nn.ReLU
        self.wid = 32
        self.model = nn.Sequential(
            nn.Linear(in_dim*self.wid*self.wid, 512),
            acti(),
            nn.Linear(512, 256),
            acti(),
            nn.Linear(256, 128),
            acti(),
            nn.Linear(128, 256),
            acti(),
            nn.Linear(256, self.wid*self.wid*self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X.view(X.shape[0], -1)).view(-1, self.out_dim, self.wid, self.wid)

class FlownetSolver():
    def __init__(self, path:str, modeltype:str, eval_only=False, smart = False, run=''):
        super().__init__()
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        print("device:",self.device)
        self.path = path
        self.run = run
        self.width = 32
        self.modeltype = modeltype
        self.discr = None
        self.r_fac = 1

        pyramid = SmartPyramid if smart else Pyramid

        if modeltype=="linear":
            self.tar_net = FullyConnected(5, 1)
            self.base_net = FullyConnected(5, 1)
            self.act_net = FullyConnected(7, 1)
            self.ext_net = FullyConnected(7, 1)
        elif modeltype=="pyramid":
            self.tar_net = pyramid(5, 1)
            self.base_net = pyramid(5, 1)
            self.act_net = pyramid(7, 1)
            self.ext_net = pyramid(7, 1)
        elif modeltype=="scnn":
            self.tar_net = FlowNet(5, 16, sequ=True, trans=False)
            self.base_net = FlowNet(5, 16, sequ=True, trans=False)
            self.act_net = FlowNet(7, 16, sequ=True, trans=False)
            self.ext_net = UpFlowNet(7, 16, sequ=True)
        elif modeltype=="brute":
            self.tar_net = pyramid(8, 1)
            self.base_net = pyramid(6, 1)
            self.act_net = pyramid(7, 1)
            self.ext_net = Discriminator(9)
        else:
            print("ERROR modeltype not understood", modeltype)

        print("succesfully initialized models")
    
    def cut_off(self, tensor, limit=1):
        tensor[tensor>limit] = limit
        return tensor

    def get_actions(self, tasks, init_scenes, brute=False):
        if brute:
            return self.brute_searched_actions(tasks, init_scenes)
        else:
            return self.generative_actions(tasks, init_scenes)
        
    def generative_actions(self, tasks, initial_scenes):
        #self.to_eval()
        actions = np.zeros((len(tasks), 3))
        num_batches = 1+len(tasks)//64
        for batch in range(num_batches):
            init_scenes = initial_scenes[batch*64:64*(batch+1)]
            #emb_init_scenes = F.embedding(T.tensor(batch_scenes), T.eye(phyre.NUM_COLORS)).transpose(-1,-3)[:,1:6].float()
            #print(init_scenes.equal(emb_init_scenes))
            with T.no_grad():
                base_paths = self.base_net(init_scenes)
                target_paths = self.tar_net(init_scenes)
                action_paths = self.act_net(T.cat((init_scenes, target_paths, base_paths), dim=1))
                action_balls = self.ext_net(T.cat((init_scenes, target_paths, action_paths), dim=1))
                print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths, action_balls), dim=1)
                #text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'base\npred', 'target\npred', 'action\npred', 'a-ball\npred']
                #vis_batch(print_batch, f'result/flownet/solver/{self.path}', f'{batch}', text=text)
            batch_task = tasks[64*batch:64*(batch+1)]
            os.makedirs(f'result/solver/generative/', exist_ok=True)
            for idx, ball in enumerate(action_balls[:,0]):
                task = batch_task[idx]

                # CHOOSE ONE VECTOR EXTRACTION METHOD
                #a = pic_to_action_vector(ball, r_fac=1.5)
                a = grow_action_vector(ball, r_fac =self.r_fac, check_border=True)
                print(a)

                drawn = draw_ball(32, *a, invert_y = True)
                pure_scene = T.as_tensor(np.max(print_batch[idx,[0,1,2,3,4]].numpy(), axis=0))
                scene_with_estimate = T.as_tensor(np.max(print_batch[idx,[0,1,2,3,4,-1]].numpy(), axis=0))
                scene_with_injected = T.as_tensor(np.max(T.stack((scene_with_estimate, drawn), dim=0).numpy(), axis=0))
                init_with_injected = T.as_tensor(np.max(T.cat((init_scenes[idx], drawn[None]), dim=0).numpy(), axis=0))
                #print(print_batch[idx].shape)
                #print(scene_with_estimate[None].shape)
                #print(scene_with_injected[None].shape)
                #print(init_with_injected[None].shape)
                pipeline = T.cat((print_batch[idx], scene_with_estimate[None], pure_scene[None], scene_with_injected[None], init_with_injected[None]), dim=0)
                
                text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'full\nscene', 'base\npred', 'target\npred', 'action\npred', 'a-ball\npred','action\nestimate', 'estimate\n/w action','   final\n   action']
                x,y,r = round(a[0], 3), round(a[1], 3), round(a[2], 3)
                vis_batch(pipeline[None], f'result/solver/generative', f"{task}__{str(a)}", text = text)
                #plt.imsave(f'result/solver/pyramid/{task}___{(x,y,r)}.png', draw_ball(32, *a, invert_y = True))
                # Radius times 4
                a[2] = a[2]*4
                #img = np.max(print_batch[idx,[0,1,2,3,4,-1]].numpy(), axis=0)
                #plt.imsave(f'result/solver/pyramid/{task}__{str(a)}.png', img)
                print(a)
                actions[idx+batch*64] = a

        #print(list(zip(tasks,actions)))

        return actions

    def generative_auccess_old(self, tasks, name, pure_noise=False):
        sim = phyre.initialize_simulator(tasks, 'ball')
        all_initial_scenes = T.tensor([[cv2.resize((scene==channel).astype(float), (32,32)) for channel in range(2,7)] for scene in sim.initial_scenes]).float().flip(-2)
        eva = phyre.Evaluator(tasks)

        self.to_train()
        actions = np.zeros((len(tasks), 3))
        pipelines = []
        num_batches = 1+len(tasks)//64
        for batch in range(num_batches):
            init_scenes = all_initial_scenes[batch*64:64*(batch+1)].to(self.device)
            #emb_init_scenes = F.embedding(T.tensor(batch_scenes), T.eye(phyre.NUM_COLORS)).transpose(-1,-3)[:,1:6].float()
            #print(init_scenes.equal(emb_init_scenes))

            # FORWARD PASS
            with T.no_grad():
                base_paths = self.base_net(init_scenes)
                target_paths = self.tar_net(init_scenes)
                action_paths = self.act_net(T.cat((init_scenes, target_paths, base_paths), dim=1))
                action_balls = self.ext_net(T.cat((init_scenes, target_paths, action_paths), dim=1))
                print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths, action_balls), dim=1)
                #text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'base\npred', 'target\npred', 'action\npred', 'a-ball\npred']
                #vis_batch(print_batch, f'result/flownet/solver/{self.path}', f'{batch}', text=text)

            batch_tasks = tasks[64*batch:64*(batch+1)]
            os.makedirs(f'result/solver/generative/', exist_ok=True)
            for idx, ball in enumerate(action_balls[:,0]):
                task = batch_tasks[idx]

                # CHOOSE ONE VECTOR EXTRACTION METHOD
                #a = pic_to_action_vector(ball, r_fac=1.5)
                mask = np.max(init_scenes[idx].cpu().numpy(), axis=0)
                #print(mask.shape)
                a = grow_action_vector(ball, r_fac =self.r_fac, num_seeds = 5, check_border=True, mask=mask)
                #print(a)

                drawn = draw_ball(32, a[0],a[1],a[2], invert_y = True).to(self.device)
                x,y,r = str(round(a[0], 2)), str(round(a[1], 2)), str(round(a[2], 2))

                """
                pure_scene = T.as_tensor(np.max(print_batch[idx,[0,1,2,3,4]].numpy(), axis=0))
                scene_with_estimate = T.as_tensor(np.max(print_batch[idx,[0,1,2,3,4,-1]].numpy(), axis=0))
                scene_with_injected = T.as_tensor(np.max(T.stack((scene_with_estimate, drawn), dim=0).numpy(), axis=0))
                init_with_injected = T.as_tensor(np.max(T.cat((init_scenes[idx], drawn[None]), dim=0).numpy(), axis=0))
                pipeline = T.cat((print_batch[idx], scene_with_estimate[None], pure_scene[None], scene_with_injected[None], init_with_injected[None]), dim=0)
                
                text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'full\nscene', 'base\npred', 'target\npred', 'action\npred', 'a-ball\npred','action\nestimate', 'estimate\n/w action','   final\n   action']
                x,y,r = round(a[0], 3), round(a[1], 3), round(a[2], 3)
                vis_batch(pipeline[None], f'result/flownet/solving/generative', f"{task}__{str(a)}", text = text)
                """
                
                back = init_scenes[None,idx,3:].sum(dim=1)[:,None]
                back = back/max(back.max(),1)
                inits = init_scenes[None,idx,None]
                vis_line = T.cat((
                    T.stack((back, inits[:,:,0]+back, inits[:,:,1]+inits[:,:,2]+back),dim=-1), # inital scene
                    T.stack((back, base_paths[idx,None]+back, inits[:,:,1]+inits[:,:,2]+back),dim=-1), # scene with base
                    T.stack((action_paths[idx,None]+back, target_paths[idx,None]+back, inits[:,:,1]+inits[:,:,2]+back),dim=-1), # scene with action path and target
                    T.stack((action_balls[idx,None]+back, inits[:,:,0]+back, inits[:,:,1]+inits[:,:,2]+back),dim=-1), #scene with action ball
                    T.stack((drawn[None,None]+back, inits[:,:,0]+back, inits[:,:,1]+inits[:,:,2]+back),dim=-1)), 
                    dim=1).detach()
                vis_line = self.cut_off(vis_line.cpu())
                white = T.ones_like(vis_line)
                white[:,:,:,:,[0,1]] -= vis_line[:,:,:,:,None,2].repeat(1,1,1,1,2)
                white[:,:,:,:,[0,2]] -= vis_line[:,:,:,:,None,1].repeat(1,1,1,1,2)
                white[:,:,:,:,[1,2]] -= vis_line[:,:,:,:,None,0].repeat(1,1,1,1,2)
                vis_line = white
                text = ['initial\nscene', 'base\nscene', 'scene\npredict', 'action\npredict', f'injected\nx {x[1:]} y {y[1:]}\nr {r[1:]}']
                vis_batch(vis_line, f'result/flownet/solving/{self.path}/{self.run}/{name}', f"{task}", text = text, save=True)
                vis_line = vis_batch(vis_line, f'result/flownet/solving/{self.path}/{self.run}/{name}', f"{task}", text = text, save=False)

                #pipelines.append(vis_line)
                
                # Radius times 4 since actions are scaled this way (and times 2 to get diameter??)
                a[2] = a[2]*4*2
                #print(a)
                # saving action
                actions[idx+batch*64] = a


                # SIMULATING ACTION
                # vis setup
                vis_count = 0
                vis_max_count = 9
                vis_wid = 64
                vis_text_actions = []
                gif_stack = T.zeros(vis_max_count+1,10,vis_wid,vis_wid, 3)

                # setup for simulation
                action = a
                delta_generator = action_delta_generator(pure_noise=pure_noise)
                task_idx = tasks.index(task)

                # First try:
                res = sim.simulate_action(task_idx, action)  
                eva.maybe_log_attempt(task_idx, res.status)

                t = 0
                warning_flag = False
                base_action = action.copy()
                while True:

                    #GIFIFY if VALID attempt
                    if not res.status.is_invalid() and vis_count<vis_max_count:
                        for i in range(min(len(res.images), 10)):
                            gif_stack[vis_count,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))
                        vis_text_actions.append(str(np.round(action, decimals=2)))
                        vis_count +=1
                    
                    # Check if SOLVED
                    if res.status.is_solved():

                        # GIFIFY
                        for i in range(min(len(res.images), 10)):
                            gif_stack[vis_max_count,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))
                        while len(vis_text_actions)<vis_max_count:
                            vis_text_actions.append('')
                        vis_text_actions.append(str(np.round(action, decimals=2))+f"\ntry {eva.attempts_per_task_index[task_idx]}")

                        print()
                        print(f"{task} solved after", eva.attempts_per_task_index[task_idx])
                        # loop untill 100 actions:
                        while eva.attempts_per_task_index[task_idx]<100:
                            eva.maybe_log_attempt(task_idx, res.status)
                    else:
                        # Try NEXT ATTEMPT:
                        if t<2000:
                            delta = delta_generator.__next__()
                            action = base_action + delta
                            print("detla:", delta, end='\r')
                            res = sim.simulate_action(task_idx, action,  need_featurized_objects=False)
                            eva.maybe_log_attempt(task_idx, res.status)
                            t += 1
                        else:
                            if not warning_flag:
                                print(f"WARNING can't find valid action for {task}")
                            warning_flag = True
                            error = True
                            eva.maybe_log_attempt(task_idx, phyre.SimulationStatus.NOT_SOLVED)
                    
                    # check if enough attempts
                    if eva.attempts_per_task_index[task_idx]>=100:
                        # GIFIFY the gif_stack
                        if not res.status.is_solved():
                            vis_text_actions.append('')
                            print()
                            print(f"{task} not solved")
                        gifify(gif_stack, f'result/flownet/solving/{self.path}/{self.run}/{name}', f'{task}_tries', text = vis_text_actions, constant=vis_line)
                        # break out and continue with next task:
                        break

        #print(list(zip(tasks,actions)))
        return eva.get_auccess()

    def generative_auccess(self, eval_setup, fold, train_mode='CONS', epochs=1):
        self.to_train()
        data_loader = self.test_dataloader
        tar_net = self.tar_net
        base_net = self.base_net
        act_net = self.act_net
        ext_net = self.ext_net

        for epoch in range(epochs):
            for i, (X,) in enumerate(data_loader):
                X = X.to(self.device)
                # Prepare Data
                action_balls = X[:,0]
                init_scenes = X[:,1:6]
                base_paths = X[:,6]
                target_paths = X[:,7]
                goal_paths = X[:,8]
                action_paths = X[:,9]

                # Optional visiualization of batch data
                #print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                #vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                # FORWARD PASS
                with T.no_grad():
                    if train_mode=='MIX':
                        modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                    else:
                        modus = train_mode

                    # Forward Pass
                    target_pred = tar_net(init_scenes)
                    base_pred = base_net(init_scenes)
                    if modus=='GT':
                        action_pred = act_net(T.cat((init_scenes, target_paths[:,None], base_paths[:,None]), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_paths[:,None], action_paths[:,None]), dim=1))
                    elif modus=='CONS':
                        action_pred = act_net(T.cat((init_scenes, target_pred.detach(), base_pred.detach()), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred.detach(), action_pred.detach()), dim=1))
                    elif modus=='COMB':
                        action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
                    elif modus=='END':
                        action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
                    

                # VISUALIZATION
                os.makedirs(f'result/flownet/inspect/{self.path}', exist_ok=True)
                Z = X.cpu()
                vis_batch(T.stack((Z, T.zeros_like(Z), T.zeros_like(Z)), dim=-1), "result/test", "color", text=["hello!"])

                print_batch = T.cat((X, base_pred, target_pred, action_pred, ball_pred), dim=1).detach()
                text = ['red', 'green', 'blue', 'blue', 'grey', 'black', 'base', 'target', 'goal\nnot used', 'action', 'base', 'target', 'action', 'red ball']
                vis_batch(print_batch.cpu(), f'result/flownet/inspect/{self.path}/{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}', text=text)

                #sum_batch = T.cat((base_paths[:,None]+base_pred, target_paths[:,None]+target_pred, action_paths[:,None]+action_pred, action_balls[:,None]+ball_pred), dim=1).detach().abs()/2
                #text = ['base\npaths', 'target\npaths', 'action\npaths', 'action\nballs']
                #vis_batch(sum_batch.cpu(), f'result/flownet/inspect/{self.path}/{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}_sum', text=text)
    
                # Extract action and draw:
                drawings = T.zeros_like(ball_pred)
                print("drawing balls for batch", i)
                for b_idx, ball in enumerate(ball_pred[:,0].cpu()):
                    a = grow_action_vector(ball, r_fac =self.r_fac, check_border=True)
                    #print(a)
                    drawn = draw_ball(32, *a, invert_y = True)
                    drawings[b_idx, 0] = drawn 

                background = init_scenes[:,3:].sum(dim=1)[:,None]
                background = background/max(background.max(), 1)
                diff_batch = T.cat((
                    T.stack((background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1), 
                    T.stack((base_pred, base_paths[:,None], T.zeros_like(base_pred)),dim=-1), 
                    T.stack((target_pred, target_paths[:,None], T.zeros_like(target_pred)),dim=-1), 
                    T.stack((action_pred, action_paths[:,None], T.zeros_like(action_pred)),dim=-1), 
                    T.stack((ball_pred, action_balls[:,None], T.zeros_like(ball_pred)),dim=-1),
                    T.stack((ball_pred+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1), 
                    T.stack((drawings+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1), 
                    T.stack((action_balls[:,None]+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1)), 
                    dim=1).detach()
                diff_bacth = self.cut_off(diff_batch.cpu())
                white = T.ones_like(diff_batch)
                white[:,:,:,:,[0,1]] -= diff_batch[:,:,:,:,None,2].repeat(1,1,1,1,2)
                white[:,:,:,:,[0,2]] -= diff_batch[:,:,:,:,None,1].repeat(1,1,1,1,2)
                white[:,:,:,:,[1,2]] -= diff_batch[:,:,:,:,None,0].repeat(1,1,1,1,2)
                diff_batch = white
                text = ['initial\nscene', 'base\npaths', 'target\npaths', 'action\npaths', 'action\nballs', 'overlay\nscene', 'injected\n scene', 'GT\nscene']
                vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/inspect/{self.path}/{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}_diff', text=text)

    def brute_searched_actions(self, tasks, all_initial_scenes):
        cache = phyre.get_default_100k_cache('ball')
        n_actions = 1000
        actions = cache.action_array[:n_actions]
        solutions = np.zeros((len(tasks), 100, 3))
        
        self.to_train()
        all_initial_scenes = all_initial_scenes.to(self.device)

        # pre-compute bases
        with T.no_grad():
            #all_base_paths = self.base_net(init_scenes)
            pass

        # loop through tasks
        for j, task in enumerate(tasks):
            # loop batched through all actions
            confs = T.zeros(n_actions)
            bs = 64
            n_batches = 1 + n_actions//bs
            #repeated_base_paths = all_base_paths[j].repeat(bs,1,1,1,1)
            repeated_initial_scenes = all_initial_scenes[j].repeat(bs,1,1,1)
            for i in range(n_batches):
                print(f"at action {i*bs} for task {task}", end='\r')
                action_batch = actions[i*bs:(i+1)*bs]
                init_scenes = repeated_initial_scenes[:len(action_batch)]
                #base_paths = repeated_base_paths[:len(action_batch)]

                # generate action pics from vectors
                action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
                for k, action_vector in enumerate(action_batch):
                    action_pics[k,0] = draw_ball(self.width, *action_vector, invert_y = True)
                action_pics = action_pics.to(self.device)

                with T.no_grad():
                    base_paths = self.base_net(T.cat((action_pics, init_scenes), dim=1))
                    action_paths = self.act_net(T.cat((action_pics, init_scenes, base_paths), dim=1))
                    target_paths = self.tar_net(T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
                    confidence = self.ext_net(T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))
                
                confs[i*bs:(i+1)*bs] = confidence[:,0]
            
            conf_args = T.argsort(confs, descending=True)
            #print(conf_args[:100])
            #print(actions[conf_args[:100]])
            best_actions = actions[conf_args[:100]]
            solutions[j] = best_actions

        return solutions

    def brute_auccess(self, tasks):    
        cache = phyre.get_default_100k_cache('ball')
        n_actions = 5000
        actions = cache.action_array[:n_actions]
        n_try = 500
        solutions = np.zeros((len(tasks), n_try, 3))

        sim = phyre.initialize_simulator(tasks, 'ball')
        all_initial_scenes = T.tensor([[cv2.resize((scene==channel).astype(float), (32,32)) for channel in range(2,7)] for scene in sim.initial_scenes]).float().flip(-2)
        eva = phyre.Evaluator(tasks)
        #vis_batch(all_initial_scenes, "result/test", "initial_scenes_vis")
        
        self.to_train()
        all_initial_scenes = all_initial_scenes.to(self.device)

        # loop through tasks
        for j, task in enumerate(tasks):
            # COLLECT 100 BEST: loop batched through all potential actions
            confs = T.zeros(n_actions)
            bs = 64
            n_batches = 1 + n_actions//bs
            #repeated_base_paths = all_base_paths[j].repeat(bs,1,1,1,1)
            repeated_initial_scenes = all_initial_scenes[j].repeat(bs,1,1,1)
            for i in range(n_batches):
                print(f"at action {i*bs} for task {task}", end='\r')
                action_batch = actions[i*bs:(i+1)*bs]
                init_scenes = repeated_initial_scenes[:len(action_batch)]
                #base_paths = repeated_base_paths[:len(action_batch)]

                # generate action pics from vectors
                action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
                for k, action_vector in enumerate(action_batch):
                    x,y,r = action_vector
                    action_pics[k,0] = draw_ball(self.width, x, y, r*0.125, invert_y = True)
                action_pics = action_pics.to(self.device)

                with T.no_grad():
                    base_paths = self.base_net(T.cat((action_pics, init_scenes), dim=1))
                    action_paths = self.act_net(T.cat((action_pics, init_scenes, base_paths), dim=1))
                    target_paths = self.tar_net(T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
                    confidence = self.ext_net(T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))
                confs[i*bs:(i+1)*bs] = confidence[:,0]
            
            # SORT OR SIMPLY TAKE GOOD ONES?
            #conf_args = T.argsort(confs, descending=True)
            #sorted_conf = T.sort(confs, descending=True)
            conf_args = T.nonzero(confs>0.9).flatten()
            sorted_conf = confs[conf_args]
            print("CONF ARGS", conf_args[:50])
            print("CONF", sorted_conf[:50])
            print("ACTIONS:", actions[conf_args[:50]])
            print("CACHE RESULTS:", cache.load_simulation_states(task)[conf_args[:50]])
            best_actions = actions[conf_args[:n_try]]
            solutions[j,:len(best_actions)] = best_actions

            # VISUALIZE best actions:
            #print(f"visualizing best actions for task {task}", end='\r')
            action_batch = best_actions
            repeated_initial_scenes = all_initial_scenes[j].repeat(len(action_batch),1,1,1)
            init_scenes = repeated_initial_scenes
            #base_paths = repeated_base_paths[:len(action_batch)]
            # generate action pics from vectors
            action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
            for k, action_vector in enumerate(action_batch):
                x,y,r = action_vector
                action_pics[k,0] = draw_ball(self.width, x, y, r*0.125, invert_y = True)
            action_pics = action_pics.to(self.device)

            with T.no_grad():
                base_paths = self.base_net(T.cat((action_pics, init_scenes), dim=1))
                action_paths = self.act_net(T.cat((action_pics, init_scenes, base_paths), dim=1))
                target_paths = self.tar_net(T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
                #init_scenes = T.cat((T.ones_like(init_scenes)[:10], init_scenes[10:]), dim=0)
                confidence = self.ext_net(T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))

                os.makedirs(f'result/flownet/solver/{self.path}/{self.run}', exist_ok=True)
                #print("conf",confidence[:20])
                print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths, T.ones_like(base_paths)*confidence[:,:,None,None]), dim=1).detach()
                text = ['red GT', 'green GT', 'blue GT', 'blue GT', 'grey GT', 'black GT', 'base\npath pred', 'target\npath pred', 'action\npath pred', 'conf']
                vis_batch(print_batch.cpu(), f'result/flownet/solving/{self.path}/{self.run}', f'{task}_best_actions', text=text)


                background = init_scenes[:,4:].sum(dim=1)[:,None]
                background = background/max(background.max(),1)
                tmp_conf = T.ones_like(base_paths)*confidence[:,:,None,None]
                scene = T.stack((action_pics+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1)
                green_target = T.stack((T.zeros_like(target_paths), target_paths, T.zeros_like(target_paths)),dim=-1)
                red_target = T.stack((action_paths, T.zeros_like(action_paths), T.zeros_like(action_paths)),dim=-1)
                diff_batch = T.cat((
                    scene,
                    scene+green_target+red_target,
                    T.stack((tmp_conf,tmp_conf,tmp_conf),dim=-1)),
                    dim=1).detach()
                diff_bacth = self.cut_off(diff_batch.cpu())
                white = T.ones_like(diff_batch)
                white[:,:,:,:,[0,1]] -= diff_batch[:,:,:,:,None,2].repeat(1,1,1,1,2)
                white[:,:,:,:,[0,2]] -= diff_batch[:,:,:,:,None,1].repeat(1,1,1,1,2)
                white[:,:,:,:,[1,2]] -= diff_batch[:,:,:,:,None,0].repeat(1,1,1,1,2)
                diff_batch = white
                text = ['scene','scene\nprediction', 'confidence']
                vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/solving/{self.path}/{self.run}', f'{task}_best_actions_diff', text=text)

            # EVALUATION
            vis_count = 0
            vis_max_count = 5
            vis_wid = 64
            vis_text_actions = []
            constant = vis_batch(self.cut_off(diff_batch.cpu()[:vis_max_count]), f'result/flownet/solving/{self.path}/{self.run}', f'{task}_best_actions_diff', text=text, save=False)
            gif_stack = T.zeros(vis_max_count+1,10,vis_wid,vis_wid, 3)
            for action in best_actions:
                if eva.attempts_per_task_index[j]>=100:
                    # GIFIFY
                    if not res.status.is_solved():
                        vis_text_actions.append('')
                        print()
                        print(f"{task} not solved")
                    gifify(gif_stack, f'result/flownet/solving/{self.path}/{self.run}', f'{task}_tries_gif', text = vis_text_actions, constant=constant)
                    break
                res = sim.simulate_action(j, action)  
                eva.maybe_log_attempt(j, res.status)

                #GIFIFY
                if not res.status.is_invalid() and vis_count<vis_max_count:
                    for i in range(min(len(res.images), 10)):
                        gif_stack[vis_count,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))
                    vis_text_actions.append(str(np.round(action, decimals=2)))
                    vis_count +=1

                if res.status.is_solved():
                    # GIFIFY
                    for i in range(min(len(res.images), 10)):
                        gif_stack[5,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))
                    while len(vis_text_actions)<vis_max_count:
                        vis_text_actions.append('')
                    vis_text_actions.append(str(np.round(action, decimals=2))+f"\ntry {eva.attempts_per_task_index[j]}")

                    print()
                    print(f"{task} solved after", eva.attempts_per_task_index[j])
                    while eva.attempts_per_task_index[j]<100:
                        eva.maybe_log_attempt(j, res.status)

        print("AUCCES", eva.get_auccess())
        return eva.get_auccess()

    def brute_proposals(self, initial_scene, actions):    
        solutions = np.zeros((len(tasks), 500, 3))
        #vis_batch(all_initial_scenes, "result/test", "initial_scenes_vis")
        
        self.to_train()
        all_initial_scenes = all_initial_scenes.to(self.device)

        # loop through tasks
        for j, task in enumerate(tasks):
            # COLLECT 100 BEST: loop batched through all potential actions
            confs = T.zeros(n_actions)
            bs = 64
            n_batches = 1 + n_actions//bs
            #repeated_base_paths = all_base_paths[j].repeat(bs,1,1,1,1)
            repeated_initial_scenes = all_initial_scenes[j].repeat(bs,1,1,1)
            for i in range(n_batches):
                print(f"at action {i*bs} for task {task}", end='\r')
                action_batch = actions[i*bs:(i+1)*bs]
                init_scenes = repeated_initial_scenes[:len(action_batch)]
                #base_paths = repeated_base_paths[:len(action_batch)]

                # generate action pics from vectors
                action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
                for k, action_vector in enumerate(action_batch):
                    x,y,r = action_vector
                    action_pics[k,0] = draw_ball(self.width, x, y, r*0.125, invert_y = True)
                action_pics = action_pics.to(self.device)

                with T.no_grad():
                    base_paths = self.base_net(T.cat((action_pics, init_scenes), dim=1))
                    action_paths = self.act_net(T.cat((action_pics, init_scenes, base_paths), dim=1))
                    target_paths = self.tar_net(T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
                    confidence = self.ext_net(T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))
                confs[i*bs:(i+1)*bs] = confidence[:,0]
            
            conf_args = T.argsort(confs, descending=True)
            sorted_conf = T.sort(confs, descending=True)
            print("CONF ARGS", conf_args[:50])
            print("CONF", sorted_conf[:50])
            print("ACTIONS:", actions[conf_args[:50]])
            print("CACHE RESULTS:", cache.load_simulation_states(task)[conf_args[:50]])
            best_actions = actions[conf_args[:500]]
            solutions[j] = best_actions

            # VISUALIZE best actions:
            #print(f"visualizing best actions for task {task}", end='\r')
            action_batch = best_actions
            repeated_initial_scenes = all_initial_scenes[j].repeat(len(action_batch),1,1,1)
            init_scenes = repeated_initial_scenes
            #base_paths = repeated_base_paths[:len(action_batch)]
            # generate action pics from vectors
            action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
            for k, action_vector in enumerate(action_batch):
                x,y,r = action_vector
                action_pics[k,0] = draw_ball(self.width, x, y, r*0.125, invert_y = True)
            action_pics = action_pics.to(self.device)

            with T.no_grad():
                base_paths = self.base_net(T.cat((action_pics, init_scenes), dim=1))
                action_paths = self.act_net(T.cat((action_pics, init_scenes, base_paths), dim=1))
                target_paths = self.tar_net(T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
                #init_scenes = T.cat((T.ones_like(init_scenes)[:10], init_scenes[10:]), dim=0)
                confidence = self.ext_net(T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))

                os.makedirs(f'result/flownet/solver/{self.path}/{self.run}', exist_ok=True)
                #print("conf",confidence[:20])
                print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths, T.ones_like(base_paths)*confidence[:,:,None,None]), dim=1).detach()
                text = ['red GT', 'green GT', 'blue GT', 'blue GT', 'grey GT', 'black GT', 'base\npath pred', 'target\npath pred', 'action\npath pred', 'conf']
                vis_batch(print_batch.cpu(), f'result/flownet/solving/{self.path}/{self.run}', f'{task}_best_actions', text=text)


                background = init_scenes[:,4:].sum(dim=1)[:,None]
                background = background/max(background.max(),1)
                tmp_conf = T.ones_like(base_paths)*confidence[:,:,None,None]
                scene = T.stack((action_pics+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1)
                green_target = T.stack((T.zeros_like(target_paths), target_paths, T.zeros_like(target_paths)),dim=-1)
                red_target = T.stack((action_paths, T.zeros_like(action_paths), T.zeros_like(action_paths)),dim=-1)
                diff_batch = T.cat((
                    scene,
                    scene+green_target+red_target,
                    T.stack((tmp_conf,tmp_conf,tmp_conf),dim=-1)),
                    dim=1).detach()
                diff_bacth = self.cut_off(diff_batch.cpu())
                white = T.ones_like(diff_batch)
                white[:,:,:,:,[0,1]] -= diff_batch[:,:,:,:,None,2].repeat(1,1,1,1,2)
                white[:,:,:,:,[0,2]] -= diff_batch[:,:,:,:,None,1].repeat(1,1,1,1,2)
                white[:,:,:,:,[1,2]] -= diff_batch[:,:,:,:,None,0].repeat(1,1,1,1,2)
                diff_batch = white
                text = ['scene','scene\nprediction', 'confidence']
                vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/solving/{self.path}/{self.run}', f'{task}_best_actions_diff', text=text)

            # EVALUATION
            vis_count = 0
            vis_max_count = 5
            vis_wid = 64
            vis_text_actions = []
            constant = vis_batch(self.cut_off(diff_batch.cpu()[:vis_max_count]), f'result/flownet/solving/{self.path}/{self.run}', f'{task}_best_actions_diff', text=text, save=False)
            gif_stack = T.zeros(vis_max_count+1,10,vis_wid,vis_wid, 3)
            for action in best_actions:
                if eva.attempts_per_task_index[j]>=100:
                    # GIFIFY
                    if not res.status.is_solved():
                        vis_text_actions.append('')
                        print()
                        print(f"{task} not solved")
                    gifify(gif_stack, f'result/flownet/solving/{self.path}/{self.run}', f'{task}_tries_gif', text = vis_text_actions, constant=constant)
                    break
                res = sim.simulate_action(j, action)  
                eva.maybe_log_attempt(j, res.status)

                #GIFIFY
                if not res.status.is_invalid() and vis_count<vis_max_count:
                    for i in range(min(len(res.images), 10)):
                        gif_stack[vis_count,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))
                    vis_text_actions.append(str(np.round(action, decimals=2)))
                    vis_count +=1

                if res.status.is_solved():
                    # GIFIFY
                    for i in range(min(len(res.images), 10)):
                        gif_stack[5,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))
                    while len(vis_text_actions)<vis_max_count:
                        vis_text_actions.append('')
                    vis_text_actions.append(str(np.round(action, decimals=2))+f"\ntry {eva.attempts_per_task_index[j]}")

                    print()
                    print(f"{task} solved after", eva.attempts_per_task_index[j])
                    while eva.attempts_per_task_index[j]<100:
                        eva.maybe_log_attempt(j, res.status)

        print("AUCCES", eva.get_auccess())
        return eva.get_auccess()

    def load_data(self, setup='ball_within_template', fold=0, train_tasks=[], test_tasks=[], brute_search=False, n_per_task=1, shuffle=True, test=False):
        fold_id = fold
        eval_setup = setup
        width = 32
        batchsize = 32

        if train_tasks and test_tasks:
            train_ids = train_tasks
            test_ids = test_tasks
            setup_name = "custom"
        else:
            setup_name = "within" if setup=='ball_within_template' else "cross"
            train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
            test_ids = dev_ids + test_ids

        if not test:
            self.train_dataloader, self.train_index = make_mono_dataset(f"data/{setup_name}_fold_{fold_id}_train_{width}xy_{n_per_task}n", 
                size=(width,width), tasks=train_ids[:], batch_size=batchsize//2 if brute_search else batchsize, n_per_task=n_per_task, shuffle=shuffle)
        else:
            self.test_dataloader, self.test_index = make_mono_dataset(f"data/{setup_name}_fold_{fold_id}_test_{width}xy_{n_per_task}n", 
                size=(width,width), tasks=test_ids, n_per_task=n_per_task, shuffle=False)
        if brute_search:
            if not test:
                self.failed_dataloader, self.failed_index = make_mono_dataset(f"data/{setup_name}_fold_{fold_id}_failed_train_{width}xy_{n_per_task}n", 
                    size=(width,width), tasks=train_ids[:], solving=False, batch_size=batchsize//2,  n_per_task=n_per_task, shuffle=shuffle)
            else:
                self.failed_test_dataloader, self.failed_test_index = make_mono_dataset(f"data/{setup_name}_fold_{fold_id}_failed_test_{width}xy_{n_per_task}n", 
                    size=(width,width), tasks=test_ids[:], solving=False, batch_size=batchsize//2,  n_per_task=n_per_task, shuffle=False)
        os.makedirs(f'result/flownet/training/{self.path}', exist_ok=True)
        with open(f'result/flownet/training/{self.path}/namespace.txt', 'w') as handle:
            handle.write(f"{self.modeltype} {setup} {fold}")

    def train_supervised(self, train_mode='CONS', epochs=10):
        self.to_train()
        data_loader = self.train_dataloader
        tar_net = self.tar_net
        base_net = self.base_net
        act_net = self.act_net
        ext_net = self.ext_net

        opti = T.optim.Adam(chain(tar_net.parameters(recurse=True), 
                            act_net.parameters(recurse=True),
                            ext_net.parameters(recurse=True),
                            base_net.parameters(recurse=True)), 
                        lr=3e-3)

        for epoch in range(epochs):
            for i, (X,) in enumerate(data_loader):
                X = X.to(self.device)
                # Prepare Data
                action_balls = X[:,0]
                init_scenes = X[:,1:6]
                base_paths = X[:,6]
                target_paths = X[:,7]
                goal_paths = X[:,8]
                action_paths = X[:,9]

                # Optional visiualization of batch data
                #print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                #vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                if train_mode=='MIX':
                    modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                else:
                    modus = train_mode

                # Forward Pass
                target_pred = tar_net(init_scenes)
                base_pred = base_net(init_scenes)
                if modus=='GT':
                    action_pred = act_net(T.cat((init_scenes, target_paths[:,None], base_paths[:,None]), dim=1))
                    ball_pred = ext_net(T.cat((init_scenes, target_paths[:,None], action_paths[:,None]), dim=1))
                elif modus=='CONS':
                    action_pred = act_net(T.cat((init_scenes, target_pred.detach(), base_pred.detach()), dim=1))
                    ball_pred = ext_net(T.cat((init_scenes, target_pred.detach(), action_pred.detach()), dim=1))
                elif modus=='COMB':
                    action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                    ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
                elif modus=='END':
                    action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                    ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
                
                if not i%30:
                    os.makedirs(f'result/flownet/training/{self.path}/{self.run}', exist_ok=True)
                    print_batch = T.cat((X, base_pred, target_pred, action_pred, ball_pred), dim=1).detach()
                    text = ['red', 'green', 'blue', 'blue', 'grey', 'black', 'base', 'target', 'goal\nnot used', 'action', 'base', 'target', 'action', 'red ball']
                    vis_batch(print_batch.cpu(), f'result/flownet/training/{self.path}/{self.run}', f'poch_{epoch}_{i}', text=text)

                    #sum_batch = T.cat((base_paths[:,None]+base_pred, target_paths[:,None]+target_pred, action_paths[:,None]+action_pred, action_balls[:,None]+ball_pred), dim=1).detach().abs()/2
                    #text = ['base\npaths', 'target\npaths', 'action\npaths', 'action\nballs']
                    #vis_batch(sum_batch.cpu(), f'result/flownet/training/{self.path}', f'poch_{epoch}_{i}_sum', text=text)

                    background = init_scenes[:,3:].sum(dim=1)[:,None]
                    background = background/max(background.max(),1)
                    diff_batch = T.cat((
                        T.stack((background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1), 
                        T.stack((base_pred, base_paths[:,None], T.zeros_like(base_pred)),dim=-1), 
                        T.stack((target_pred, target_paths[:,None], T.zeros_like(target_pred)),dim=-1), 
                        T.stack((action_pred, action_paths[:,None], T.zeros_like(action_pred)),dim=-1), 
                        T.stack((ball_pred, action_balls[:,None], T.zeros_like(ball_pred)),dim=-1),
                        T.stack((ball_pred+background, init_scenes[:,None,1]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1), 
                        T.stack((action_balls[:,None]+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1)), 
                        dim=1).detach()
                    diff_bacth = self.cut_off(diff_batch.cpu())
                    white = T.ones_like(diff_batch)
                    white[:,:,:,:,[0,1]] -= diff_batch[:,:,:,:,None,2].repeat(1,1,1,1,2)
                    white[:,:,:,:,[0,2]] -= diff_batch[:,:,:,:,None,1].repeat(1,1,1,1,2)
                    white[:,:,:,:,[1,2]] -= diff_batch[:,:,:,:,None,0].repeat(1,1,1,1,2)
                    diff_batch = white
                    text = ['initial\nscene', 'base\npaths', 'target\npaths', 'action\npaths', 'action\nballs', 'injected\nscene', 'GT\nscene']
                    vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/training/{self.path}/{self.run}', f'poch_{epoch}_{i}_diff', text=text)
                #plt.show()

                # Loss
                tar_loss = F.binary_cross_entropy(target_pred, target_paths[:,None])
                act_loss = F.binary_cross_entropy(action_pred, action_paths[:,None])
                ball_loss = F.binary_cross_entropy(ball_pred, action_balls[:,None])
                base_loss = F.binary_cross_entropy(base_pred, base_paths[:,None])
                if modus=='END':
                    loss = ball_loss
                else:
                    loss = ball_loss + tar_loss + act_loss + base_loss
                print(epoch, i, loss.item(), end='\r')

                # Backward Pass
                opti.zero_grad()
                loss.backward()
                opti.step()
    
    def train_brute_search(self, train_mode='CONS', epochs=10):
        self.to_train()
        data_loader = self.train_dataloader
        fail_loader = self.failed_dataloader
        base_net = self.base_net
        act_net = self.act_net
        tar_net = self.tar_net
        ext_net = self.ext_net

        opti = T.optim.Adam(chain(tar_net.parameters(recurse=True), 
                            act_net.parameters(recurse=True),
                            ext_net.parameters(recurse=True),
                            base_net.parameters(recurse=True)), 
                        lr=3e-3)

        for epoch in range(epochs):
            for i, ((X,), (Z,)) in enumerate(zip(data_loader, fail_loader)):
                last_index = min(X.shape[0], Z.shape[0])
                X, Z = X[:last_index].to(self.device), Z[:last_index].to(self.device)

                # Prepare Data
                solve_scenes = X[:,:6]
                solve_base_paths = X[:,6]
                solve_target_paths = X[:,7]
                solve_action_paths = X[:,9]
                fail_scenes = Z[:,[0,1,2,3,4,5]]
                fail_base_paths = Z[:,6]
                fail_target_paths = Z[:,7]
                fail_action_paths = Z[:,9]

                init_scenes = T.cat((solve_scenes, fail_scenes), dim=0)
                target_paths = T.cat((solve_target_paths, fail_target_paths), dim=0)
                base_paths = T.cat((solve_base_paths, fail_base_paths), dim=0)
                action_paths = T.cat((solve_action_paths, fail_action_paths), dim=0)

                # Optional visiualization of batch data
                #print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                #vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                if train_mode=='MIX':
                    modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                else:
                    modus = train_mode

                # Forward Pass
                base_pred = base_net(init_scenes)
                if modus=='GT':
                    action_pred = act_net(T.cat((init_scenes, target_paths[:,None], base_paths[:,None]), dim=1))
                    confidence = ext_net(T.cat((init_scenes, target_paths[:,None], action_paths[:,None]), dim=1))
                elif modus=='CONS':
                    action_pred = act_net(T.cat((init_scenes, base_pred.detach()), dim=1))
                    target_pred = tar_net(T.cat((init_scenes, base_pred.detach(), action_pred.detach()), dim=1))
                    confidence = ext_net(T.cat((init_scenes, base_pred.detach(), target_pred.detach(), action_pred.detach()), dim=1))
                elif modus=='COMB':
                    action_pred = act_net(T.cat((init_scenes, base_pred), dim=1))
                    target_pred = tar_net(T.cat((init_scenes, base_pred, action_pred), dim=1))
                    confidence = ext_net(T.cat((init_scenes, base_pred, target_pred, action_pred), dim=1))
                elif modus=='END':
                    action_pred = act_net(T.cat((init_scenes, base_pred), dim=1))
                    target_pred = tar_net(T.cat((init_scenes, base_pred, action_pred), dim=1))
                    confidence = ext_net(T.cat((init_scenes, base_pred, target_pred, action_pred), dim=1))
                
                if not i%30:
                    os.makedirs(f'result/flownet/training/{self.path}', exist_ok=True)
                    print_batch = T.cat((T.cat((X,Z), dim=0), base_pred, target_pred, action_pred, T.ones_like(base_pred)*confidence[:,:,None,None]), dim=1).detach()
                    text = ['red GT', 'green GT', 'blue GT', 'blue GT', 'grey GT', 'black GT', 'base\nGT path', 'target\nGT path', 'goal GT\nnot used', 'action\nGT path', 'base\npath pred', 'target\npath pred', 'action\npath pred', 'conf']
                    vis_batch(print_batch.cpu(), f'result/flownet/training/{self.path}/{self.run}', f'poch_{epoch}_{i}', text=text)

                    background = init_scenes[:,4:].sum(dim=1)[:,None]
                    background = background/max(background.max(),1)
                    tmp_conf = T.ones_like(base_pred)*confidence[:,:,None,None]
                    diff_batch = T.cat((
                        T.stack((init_scenes[:,None,0]+background, init_scenes[:,None,1]+background, init_scenes[:,None,2]+init_scenes[:,None,3]+background),dim=-1), 
                        T.stack((base_pred, base_paths[:,None], T.zeros_like(base_pred)),dim=-1), 
                        T.stack((target_pred, target_paths[:,None], T.zeros_like(target_pred)),dim=-1), 
                        T.stack((action_pred, action_paths[:,None], T.zeros_like(action_pred)),dim=-1), 
                        T.stack((tmp_conf,tmp_conf,tmp_conf),dim=-1)),
                        dim=1).detach()
                    diff_bacth = self.cut_off(diff_batch.cpu())
                    white = T.ones_like(diff_batch)
                    white[:,:,:,:,[0,1]] -= diff_batch[:,:,:,:,None,2].repeat(1,1,1,1,2)
                    white[:,:,:,:,[0,2]] -= diff_batch[:,:,:,:,None,1].repeat(1,1,1,1,2)
                    white[:,:,:,:,[1,2]] -= diff_batch[:,:,:,:,None,0].repeat(1,1,1,1,2)
                    diff_batch = white
                    text = ['scene','base\npaths', 'target\npaths', 'action\npaths', 'confidence']
                    vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/training/{self.path}/{self.run}', f'poch_{epoch}_{i}_diff', text=text)
                #plt.show()

                # Loss
                tar_loss = F.binary_cross_entropy(target_pred, target_paths[:,None])
                act_loss = F.binary_cross_entropy(action_pred, action_paths[:,None])
                conf_target = T.cat((T.ones(X.shape[0]), T.zeros(Z.shape[0])), dim=0).to(self.device)
                conf_loss = F.binary_cross_entropy(confidence[:,0], conf_target)
                base_loss = F.binary_cross_entropy(base_pred, base_paths[:,None])

                print(epoch, i,(confidence[:,0].round()==conf_target).float().sum().item()/confidence.shape[0], "classified correctly", end='\r')

                if modus=='END':
                    loss = conf_loss
                else:
                    loss = conf_loss + tar_loss + act_loss + base_loss
                #print(epoch, i, loss.item(), end='\r')

                # Backward Pass
                opti.zero_grad()
                loss.backward()
                opti.step()

    def inspect_supervised(self, eval_setup, fold, train_mode='CONS', epochs=1):
        self.to_train()
        data_loader = self.test_dataloader
        tar_net = self.tar_net
        base_net = self.base_net
        act_net = self.act_net
        ext_net = self.ext_net

        for epoch in range(epochs):
            for i, (X,) in enumerate(data_loader):
                if i>10:
                    break
                X = X.to(self.device)
                # Prepare Data
                action_balls = X[:,0]
                init_scenes = X[:,1:6]
                base_paths = X[:,6]
                target_paths = X[:,7]
                goal_paths = X[:,8]
                action_paths = X[:,9]

                # Optional visiualization of batch data
                #print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                #vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                # FORWARD PASS
                with T.no_grad():
                    if train_mode=='MIX':
                        modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                    else:
                        modus = train_mode

                    # Forward Pass
                    target_pred = tar_net(init_scenes)
                    base_pred = base_net(init_scenes)
                    if modus=='GT':
                        action_pred = act_net(T.cat((init_scenes, target_paths[:,None], base_paths[:,None]), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_paths[:,None], action_paths[:,None]), dim=1))
                    elif modus=='CONS':
                        action_pred = act_net(T.cat((init_scenes, target_pred.detach(), base_pred.detach()), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred.detach(), action_pred.detach()), dim=1))
                    elif modus=='COMB':
                        action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
                    elif modus=='END':
                        action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
                    

                # VISUALIZATION
                os.makedirs(f'result/flownet/inspect/{self.path}/{self.run}', exist_ok=True)
                #Z = X.cpu()
                #vis_batch(T.stack((Z, T.zeros_like(Z), T.zeros_like(Z)), dim=-1), "result/test", "color", text=["hello!"])

                print_batch = T.cat((X, base_pred, target_pred, action_pred, ball_pred), dim=1).detach()
                text = ['red', 'green', 'blue', 'blue', 'grey', 'black', 'base', 'target', 'goal\nnot used', 'action', 'base', 'target', 'action', 'red ball']
                vis_batch(print_batch.cpu(), f'result/flownet/inspect/{self.path}/{self.run}/{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}', text=text)

                #sum_batch = T.cat((base_paths[:,None]+base_pred, target_paths[:,None]+target_pred, action_paths[:,None]+action_pred, action_balls[:,None]+ball_pred), dim=1).detach().abs()/2
                #text = ['base\npaths', 'target\npaths', 'action\npaths', 'action\nballs']
                #vis_batch(sum_batch.cpu(), f'result/flownet/inspect/{self.path}/{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}_sum', text=text)
    
                # Extract action and draw:
                drawings = T.zeros_like(ball_pred)
                print("drawing balls for batch", i)
                for b_idx, ball in enumerate(ball_pred[:,0].cpu()):
                    a = grow_action_vector(ball, r_fac =self.r_fac, check_border=True)
                    #print(a)
                    drawn = draw_ball(32, a[0],a[1],a[2], invert_y = True)
                    drawings[b_idx, 0] = drawn 

                background = init_scenes[:,3:].sum(dim=1)[:,None]
                background = background/max(background.max(),1)
                diff_batch = T.cat((
                    T.stack((background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1), 
                    T.stack((base_pred, base_paths[:,None], T.zeros_like(base_pred)),dim=-1), 
                    T.stack((target_pred, target_paths[:,None], T.zeros_like(target_pred)),dim=-1), 
                    T.stack((action_pred, action_paths[:,None], T.zeros_like(action_pred)),dim=-1), 
                    T.stack((ball_pred, action_balls[:,None], T.zeros_like(ball_pred)),dim=-1),
                    T.stack((ball_pred+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1), 
                    T.stack((drawings+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1), 
                    T.stack((action_balls[:,None]+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1)), 
                    dim=1).detach()
                diff_bacth = self.cut_off(diff_batch.cpu())
                white = T.ones_like(diff_batch)
                white[:,:,:,:,[0,1]] -= diff_batch[:,:,:,:,None,2].repeat(1,1,1,1,2)
                white[:,:,:,:,[0,2]] -= diff_batch[:,:,:,:,None,1].repeat(1,1,1,1,2)
                white[:,:,:,:,[1,2]] -= diff_batch[:,:,:,:,None,0].repeat(1,1,1,1,2)
                diff_batch = white
                text = ['initial\nscene', 'base\npaths', 'target\npaths', 'action\npaths', 'action\nballs', 'overlay\nscene', 'injected\n scene', 'GT\nscene']
                vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/inspect/{self.path}/{self.run}/{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}_diff', text=text)

    def inspect_brute_search(self, eval_setup, fold, train_mode='CONS', epochs=1):
        self.to_train()
        data_loader = self.test_dataloader
        fail_loader = self.failed_test_dataloader
        base_net = self.base_net
        act_net = self.act_net
        tar_net = self.tar_net
        ext_net = self.ext_net

        percentage = []

        for epoch in range(epochs):
            for i, ((X,), (Z,)) in enumerate(zip(data_loader, fail_loader)):
                last_index = min(X.shape[0], Z.shape[0])
                X, Z = X[:last_index].to(self.device), Z[:last_index].to(self.device)

                # Prepare Data
                solve_scenes = X[:,:6]
                solve_base_paths = X[:,6]
                solve_target_paths = X[:,7]
                solve_action_paths = X[:,9]
                fail_scenes = Z[:,[0,1,2,3,4,5]]
                fail_base_paths = Z[:,6]
                fail_target_paths = Z[:,7]
                fail_action_paths = Z[:,9]

                init_scenes = T.cat((solve_scenes, fail_scenes), dim=0)
                target_paths = T.cat((solve_target_paths, fail_target_paths), dim=0)
                base_paths = T.cat((solve_base_paths, fail_base_paths), dim=0)
                action_paths = T.cat((solve_action_paths, fail_action_paths), dim=0)

                # Optional visiualization of batch data
                #print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                #vis_batch(X, f'data/flownet', f'{epoch}_{i}')
                with T.no_grad():
                    if train_mode=='MIX':
                        modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                    else:
                        modus = train_mode

                    # Forward Pass
                    base_pred = base_net(init_scenes)
                    if modus=='GT':
                        action_pred = act_net(T.cat((init_scenes, target_paths[:,None], base_paths[:,None]), dim=1))
                        confidence = ext_net(T.cat((init_scenes, target_paths[:,None], action_paths[:,None]), dim=1))
                    elif modus=='CONS':
                        action_pred = act_net(T.cat((init_scenes, base_pred.detach()), dim=1))
                        target_pred = tar_net(T.cat((init_scenes, base_pred.detach(), action_pred.detach()), dim=1))
                        confidence = ext_net(T.cat((init_scenes, base_pred.detach(), target_pred.detach(), action_pred.detach()), dim=1))
                    elif modus=='COMB':
                        action_pred = act_net(T.cat((init_scenes, base_pred), dim=1))
                        target_pred = tar_net(T.cat((init_scenes, base_pred, action_pred), dim=1))
                        confidence = ext_net(T.cat((init_scenes, base_pred, target_pred, action_pred), dim=1))
                    elif modus=='END':
                        action_pred = act_net(T.cat((init_scenes, base_pred), dim=1))
                        target_pred = tar_net(T.cat((init_scenes, base_pred, action_pred), dim=1))
                        confidence = ext_net(T.cat((init_scenes, base_pred, target_pred, action_pred), dim=1))

                # EVAL PERFORMANCE:
                conf_target = T.cat((T.ones(X.shape[0]), T.zeros(Z.shape[0])), dim=0).to(self.device)
                percentage.append((confidence[:,0].round()==conf_target).float().sum().item()/confidence.shape[0])
                print("correct classified percentage:", percentage[-1], end='\r')
                   
                os.makedirs(f'result/flownet/training/{self.path}/{self.run}', exist_ok=True)
                print_batch = T.cat((T.cat((X,Z), dim=0), base_pred, target_pred, action_pred, T.ones_like(base_pred)*confidence[:,:,None,None]), dim=1).detach()
                text = ['red', 'green', 'blue', 'blue', 'grey', 'black', 'base', 'target', 'goal\nnot used', 'action', 'base', 'target', 'action', 'conf']
                vis_batch(print_batch.cpu(), f'result/flownet/inspect/{self.path}/{self.run}/brute_{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}', text=text)

                background = init_scenes[:,4:].sum(dim=1)[:,None]
                background = background/max(background.max(),1)
                tmp_conf = T.ones_like(base_pred)*confidence[:,:,None,None]
                diff_batch = T.cat((
                    T.stack((init_scenes[:,None,0]+background, init_scenes[:,None,1]+background, init_scenes[:,None,2]+init_scenes[:,None,3]+background),dim=-1), 
                    T.stack((base_pred, base_paths[:,None], T.zeros_like(base_pred)),dim=-1), 
                    T.stack((target_pred, target_paths[:,None], T.zeros_like(target_pred)),dim=-1), 
                    T.stack((action_pred, action_paths[:,None], T.zeros_like(action_pred)),dim=-1), 
                    T.stack((tmp_conf,tmp_conf,tmp_conf),dim=-1)),
                    dim=1).detach()
                diff_bacth = self.cut_off(diff_batch.cpu())
                white = T.ones_like(diff_batch)
                white[:,:,:,:,[0,1]] -= diff_batch[:,:,:,:,None,2].repeat(1,1,1,1,2)
                white[:,:,:,:,[0,2]] -= diff_batch[:,:,:,:,None,1].repeat(1,1,1,1,2)
                white[:,:,:,:,[1,2]] -= diff_batch[:,:,:,:,None,0].repeat(1,1,1,1,2)
                diff_batch = white
                text = ['scene','base\npaths', 'target\npaths', 'action\npaths', 'confidence']
                vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/inspect/{self.path}/{self.run}/brute_{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}_diff', text=text)

        print()
        print("avg correct classified percentage:", sum(percentage)/len(percentage))
        print(percentage)
        os.makedirs(f'result/solver/result/{self.path}/{self.run}', exist_ok=True)
        with open(f'result/solver/result/{self.path}/{self.run}/accuracy_{eval_setup}_fold_{fold}.txt', 'a') as fp:
            fp.write(f"avg correct classified percentage: {sum(percentage)/len(percentage)}")


    def to_eval(self):
        self.tar_net = self.tar_net.cpu()
        self.base_net = self.base_net.cpu()
        self.act_net = self.act_net.cpu()
        self.ext_net = self.ext_net.cpu()
        self.tar_net.eval()
        self.base_net.eval()
        self.act_net.eval()
        self.ext_net.eval()
    
    def to_train(self):
        if self.device=="cuda":
            self.tar_net = self.tar_net.cuda()
            self.base_net = self.base_net.cuda()
            self.act_net = self.act_net.cuda()
            self.ext_net = self.ext_net.cuda()
        self.tar_net.train()
        self.base_net.train()
        self.act_net.train()
        self.ext_net.train()

    def load_models(self, setup="ball_within_template", fold=0, device='cpu'):
        setup_name = "within" if setup=='ball_within_template' else ("cross"  if setup=='ball_cross_template' else "custom")

        self.tar_net.load_state_dict(T.load(f"saves/flownet/{self.path}_tar_{setup_name}_{fold}.pt", map_location=T.device(device)))
        self.act_net.load_state_dict(T.load(f"saves/flownet/{self.path}_act_{setup_name}_{fold}.pt", map_location=T.device(device)))
        self.ext_net.load_state_dict(T.load(f"saves/flownet/{self.path}_ext_{setup_name}_{fold}.pt", map_location=T.device(device)))
        self.base_net.load_state_dict(T.load(f"saves/flownet/{self.path}_base_{setup_name}_{fold}.pt", map_location=T.device(device)))
        if self.discr is not None:
            self.discr.load_state_dict(T.load(f"saves/flownet/{self.path}_discr_{setup_name}_{fold}.pt", map_location=T.device(device)))

    def save_models(self, setup="ball_within_template", fold=0):
        setup_name = "within" if setup=='ball_within_template' else ("cross"  if setup=='ball_cross_template' else "custom")

        T.save(self.tar_net.state_dict(), f"saves/flownet/{self.path}_tar_{setup_name}_{fold}.pt")
        T.save(self.act_net.state_dict(), f"saves/flownet/{self.path}_act_{setup_name}_{fold}.pt")
        T.save(self.ext_net.state_dict(), f"saves/flownet/{self.path}_ext_{setup_name}_{fold}.pt")
        T.save(self.base_net.state_dict(), f"saves/flownet/{self.path}_base_{setup_name}_{fold}.pt")
        if self.discr is not None:
            T.save(self.discr.state_dict(), f"saves/flownet/{self.path}_discr_{setup_name}_{fold}.pt")

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
#PARSER SETUP
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_id', type=str)
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-trans', action='store_true')
    parser.add_argument('-sequ', action='store_true')
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-gan', action='store_true')
    parser.add_argument('-linear', action='store_true')
    parser.add_argument('-pyramid', action='store_true')
    parser.add_argument('--train_mode', default='GT', type=str, choices=['GT', 'MIX', 'CONS', 'COMB', 'END'])
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--width', default=32, type=int)
    parser.add_argument('--visevery', default=10, type=int)
    args = parser.parse_args()
    print(args)
#%%
# DATA SETUP
if __name__ == "__main__":
    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    if args.train:
        train_dataloader, index = make_mono_dataset(f"data/phyre_fold_{fold_id}_train_{args.width}", size=(args.width,args.width), tasks=train_ids[:])
        os.makedirs(f'result/flownet/training/{args.path_id}', exist_ok=True)
        with open(f'result/flownet/training/{args.path_id}/namespace.txt', 'w') as handle:
            handle.write(str(args))
#%%
# NETS SETUP
if __name__ == "__main__":
    if args.linear:
        tar_net = FullyConnected(5, 1)
        base_net = FullyConnected(5, 1)
        act_net = FullyConnected(7, 1)
        ext_net = FullyConnected(7, 1)
    elif args.pyramid:
        tar_net = Pyramid(5, 1)
        base_net = Pyramid(5, 1)
        act_net = Pyramid(7, 1)
        ext_net = Pyramid(7, 1)
    else:
        tar_net = FlowNet(5, 16, sequ=args.sequ, trans=args.trans)
        base_net = FlowNet(5, 16, sequ=args.sequ, trans=args.trans)
        act_net = FlowNet(7, 16, sequ=args.sequ, trans=args.trans)
        ext_net = UpFlowNet(7, 16, sequ=args.sequ)
    discr = Discriminator(8)
#%%
# OPTI SETUP
if __name__ == "__main__" and args.train:
    #opti = T.optim.Adam(tar_net.parameters(recurse=True), lr=1e-3)
    #opti2 = T.optim.Adam(act_net.parameters(recurse=True), lr=1e-3)
    #opti3 = T.optim.Adam(ext_net.parameters(recurse=True), lr=1e-3)
    nets_opti = T.optim.Adam(chain(tar_net.parameters(recurse=True), 
                            act_net.parameters(recurse=True),
                            ext_net.parameters(recurse=True),
                            base_net.parameters(recurse=True)), 
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

def train_supervised(epochs:int, base_net:FlowNet, tar_net:FlowNet, act_net:FlowNet, ext_net:UpFlowNet,
    data_loader:T.utils.data.DataLoader, opti: T.optim.Adam, train_mode='GT'):
    for epoch in range(epochs):
        for i, (X,) in enumerate(data_loader):
            # Prepare Data
            action_balls = X[:,0]
            init_scenes = X[:,1:6]
            base_paths = X[:,6]
            target_paths = X[:,7]
            goal_paths = X[:,8]
            action_paths = X[:,9]

            # Optional visiualization of batch data
            #print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
            #vis_batch(X, f'data/flownet', f'{epoch}_{i}')

            if train_mode=='MIX':
                modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
            else:
                modus = train_mode

            # Forward Pass
            target_pred = tar_net(init_scenes)
            base_pred = base_net(init_scenes)
            if modus=='GT':
                action_pred = act_net(T.cat((init_scenes, target_paths[:,None], base_paths[:,None]), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_paths[:,None], action_paths[:,None]), dim=1))
            elif modus=='CONS':
                action_pred = act_net(T.cat((init_scenes, target_pred.detach(), base_pred.detach()), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_pred.detach(), action_pred.detach()), dim=1))
            elif modus=='COMB':
                action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
            elif modus=='END':
                action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
            
            if not i%10:
                os.makedirs(f'result/flownet/training/{args.path_id}', exist_ok=True)
                print_batch = T.cat((X, base_pred, target_pred, action_pred, ball_pred), dim=1).detach()
                vis_batch(print_batch, f'result/flownet/training/{args.path_id}', f'poch_{epoch}_{i}')
            #plt.show()

            # Loss
            tar_loss = F.binary_cross_entropy(target_pred, target_paths[:,None])
            act_loss = F.binary_cross_entropy(action_pred, action_paths[:,None])
            ball_loss = F.binary_cross_entropy(ball_pred, action_balls[:,None])
            base_loss = F.binary_cross_entropy(base_pred, base_paths[:,None])
            if modus=='END':
                loss = ball_loss
            else:
                loss = ball_loss + tar_loss + act_loss + base_loss
            print(epoch, i, loss.item())

            # Backward Pass
            opti.zero_grad()
            loss.backward()
            opti.step()

def train_as_gan(epochs:int, tar_net:FlowNet, act_net:FlowNet, ext_net:UpFlowNet, discr:Discriminator,
    data_loader:T.utils.data.DataLoader, nets_opti: T.optim.Optimizer, discr_opti: T.optim.Optimizer, use_GT=True):
    switch = False
    for epoch in range(epochs):
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
                    os.makedirs(f'result/flownet/training/{args.path_id}', exist_ok=True)
                    print_batch = T.cat((X, target_pred, action_pred, ball_pred), dim=1).detach()
                    vis_batch(print_batch, f'result/flownet/training/{args.path_id}', f'poch_{epoch}_{i}')
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
if __name__ == "__main__" and args.train:
    #train()
    #model.load_state_dict(T.load("saves/imaginet-c16-all.pt"))
    #train2()
    #model2.load_state_dict(T.load("saves/imaginet2-c16-all.pt"))
    if args.gan:
        train_as_gan(args.epochs, tar_net, act_net, ext_net, discr, train_dataloader, nets_opti, discr_opti, train_mode=args.train_mode)
    else:
        train_supervised(args.epochs, base_net, tar_net, act_net, ext_net, train_dataloader, nets_opti, train_mode=args.train_mode)

# %%
if __name__ == "__main__" and args.train:
    os.makedirs(f"saves/flownet/", exist_ok=True)
    T.save(tar_net.state_dict(), f"saves/flownet/flownet_tar_{args.path_id}.pt")
    T.save(act_net.state_dict(), f"saves/flownet/flownet_act_{args.path_id}.pt")
    T.save(ext_net.state_dict(), f"saves/flownet/flownet_ext_{args.path_id}.pt")
    T.save(base_net.state_dict(), f"saves/flownet/flownet_base_{args.path_id}.pt")
    if args.gan:
        T.save(discr.state_dict(), f"saves/flownet_discr_{args.path_id}.pt")

#%%
def inspect(tar_net:FlowNet, act_net:FlowNet, ext_net:UpFlowNet, 
    data_loader:T.utils.data.DataLoader):
    os.makedirs(f'result/flownet/testing/{args.path_id}', exist_ok=True)
    with open(f'result/flownet/testing/{args.path_id}/namespace.txt', 'w') as handle:
        handle.write(str(args))

    for i, (X,) in enumerate(data_loader):
        init_scenes = X[:,1:6]
        target_paths = X[:,6]
        action_paths = X[:,8]
        goal_paths = X[:,7]
        base_paths = X[:,9]
        action_balls = X[:,0]

        with T.no_grad():
            base_pred = base_net(init_scenes)
            target_pred = tar_net(init_scenes)
            action_pred = act_net(T.cat((init_scenes, target_pred, base_paths[:,None]), dim=1))
            ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))

        print_batch = T.cat((X, base_pred, target_pred, action_pred, ball_pred), dim=1).detach()
        vis_batch(print_batch, f'result/flownet/testing/{args.path_id}', 'batch_{i}')

if __name__ == "__main__" and args.test:
    fold_id=0
    tar_net.load_state_dict(T.load(f"saves/flownet/flownet_tar_{args.path_id}.pt"))
    act_net.load_state_dict(T.load(f"saves/flownet/flownet_act_{args.path_id}.pt"))
    ext_net.load_state_dict(T.load(f"saves/flownet/flownet_ext_{args.path_id}.pt"))
    base_net.load_state_dict(T.load(f"saves/flownet/flownet_base_{args.path_id}.pt"))
    test_dataloader, index = make_mono_dataset(f"data/phyre_fold_{fold_id}_test_32", size=(32,32), tasks=test_ids)
    inspect(tar_net, act_net, ext_net, test_dataloader)
# %%
