#%%
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import MNIST
import torchvision as TV
from matplotlib import pyplot as plt
from IPython.display import clear_output
from itertools import chain
from tqdm import tqdm
#%%
class InfoEncoder(nn.Module):
    def __init__(self, feature_dims, embed_dims):
        super().__init__()
        self.feature_dims = feature_dims
        self.embed_dims = embed_dims
        self.lin_enc1 = nn.Sequential(
            nn.Linear(feature_dims,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dims),       
            nn.Tanh()
        )
        self.lin_enc2 = nn.Sequential(
            nn.Linear(feature_dims,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dims),       
            nn.Tanh()
        )
        self.lin_dec = nn.Sequential(
            nn.Linear(embed_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dims),
            nn.Tanh()
        )
    
    def train(self, X: T.Tensor, opti):
        X = X.flatten(start_dim=1)
        emb1 = self.lin_enc1(X)
        res = self.lin_dec(emb1)
        target_loss = F.mse_loss(res, X)
        #emb2 = self.lin_enc2(res)
        #emb_loss = F.mse_loss(emb2, emb1)
        loss = target_loss #+ emb_loss
        opti.zero_grad()
        loss.backward()
        opti.step()
        return res.detach().view(-1,1,28,28), target_loss.item(), 0#emb_loss.item()




#%%
inf_enc = InfoEncoder(28**2, 16)
#%%
if __name__ == "__main__":
    batch_size = 32
    data_loader = T.utils.data.DataLoader(
        MNIST('data', train=True, download=True, 
        transform=TV.transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    opti = T.optim.Adam(chain(
        inf_enc.lin_dec.parameters(),
        inf_enc.lin_enc1.parameters(),
        inf_enc.lin_enc2.parameters()),
        lr=1e-2
    )
    
    for i, (X, Y) in tqdm(enumerate(data_loader)):
        clear_output(wait=True)
        grid = TV.utils.make_grid(X[:16])
        plt.imshow(grid.transpose(0,2).transpose(0,1))
        plt.show()
        res, tl, el = inf_enc.train(X, opti)
        print(tl, el)
        grid = TV.utils.make_grid(res[:16])
        plt.imshow(grid.transpose(0,2).transpose(0,1))
        plt.show()

# %%
T.save(inf_enc, 'inf_enc.pt')


# %%
