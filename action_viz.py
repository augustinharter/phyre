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

width = 32
train_ids = []
batchsize = 64
brute_search = False
n_per_task = 10
shuffle = True
data_loader, index = make_mono_dataset(f"data/within_fold_0_train_32xy_10n", 
                size=(width,width), tasks=train_ids[:], batch_size=batchsize//2 if brute_search else batchsize, n_per_task=n_per_task, shuffle=shuffle)

for i, (X,) in enumerate(data_loader):
    init_scenes = X[:,1:6]
    action_balls = X[:,None,0]
    drawn = T.zeros_like(action_balls)
    action_list = []
    #print(mask.shape)
    for idx,ball in enumerate(action_balls):
        print(idx)
        mask = np.max(init_scenes[idx].cpu().numpy(), axis=0)
        a = grow_action_vector(ball[0], r_fac =1, num_seeds = 5, check_border=True, mask=mask)
        local_draw = draw_ball(32, a[0],a[1],a[2], invert_y = True)
        x,y,r = str(round(a[0], 2)), str(round(a[1], 2)), str(round(a[2], 2))
        action_list.append((x,y,r))
        drawn[idx] = local_draw

    back = init_scenes[:,3:].sum(dim=1)[:,None]
    back = back/max(back.max(),1)
    inits = init_scenes[:,None]
    vis_line = T.cat((
        T.stack((action_balls+back, inits[:,:,0]+back, inits[:,:,1]+inits[:,:,2]+back),dim=-1), # inital scene
        T.stack((drawn+back, inits[:,:,0]+back, inits[:,:,1]+inits[:,:,2]+back),dim=-1)), 
        dim=1).detach()
    vis_line[vis_line>1] = 1
    white = T.ones_like(vis_line)
    white[:,:,:,:,[0,1]] -= vis_line[:,:,:,:,None,2].repeat(1,1,1,1,2)
    white[:,:,:,:,[0,2]] -= vis_line[:,:,:,:,None,1].repeat(1,1,1,1,2)
    white[:,:,:,:,[1,2]] -= vis_line[:,:,:,:,None,0].repeat(1,1,1,1,2)
    vis_line = white
    text = ['initial\nscene', f'injected']
    vis_batch(vis_line, f'result/test/action_viz/', f"{i}", text = text, save=True)
    #vis_line = vis_batch(vis_line, f'result/flownet/solving/{self.path}/{self.run}/{name}', f"{task}", text = text, save=False)
