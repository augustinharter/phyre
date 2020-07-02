from imaginet import *
import torch as T
import phyre
import numpy as np
import cv2

def solve(model, model2):
    tasks = ['00000:001', '00000:002', '00000:003', '00000:004', '00000:005',
            '00001:001', '00001:002', '00001:003', '00001:004', '00001:005',
            '00002:007', '00002:011', '00002:015', '00002:017', '00002:023',
            '00003:000', '00003:001', '00003:002', '00003:003', '00003:004',
            '00004:063', '00004:071', '00004:092', '00004:094', '00004:095']
    sim = phyre.initialize_simulator(tasks, 'ball')

    init_scenes = sim.initial_scenes
    X = T.tensor(format(init_scenes)).float()
    print(X.shape)

    batch = []
    for i in range(len(tasks)):
        while True:
            action = sim.sample(i)
            action[2] = 0.01
            res = sim.simulate_action(i, action, stride=20)
            if type(res.images)!=type(None):
                batch.append(res.images)
                break
    base = T.tensor(rollouts_to_base(batch)).float()

    with T.no_grad():
        Z = model(X)
        print(base.shape, Z.shape)
        A = model2(T.cat((X[:,1:], base[:,None], Z), dim=1))
    #B = model3(T.cat((X[:,1:], Y[:,None,2], Z, A), dim=1))
    B = extract_action(A)
    print(B.shape)
    gen_actions = []
    for b in B[:,0]:
        gen_actions.append(pic_to_values(b))
    print(gen_actions)
    #TODO feed actions into simulator

def rollouts_to_base(batch, size=(32,32)):
    B = np.zeros((len(batch), size[0], size[1]))
    for j, r in enumerate(batch):
        path = np.zeros((len(r), size[0], size[1]))
        for i, scene in enumerate(r):
            a_chan = (scene==j).astype(float)
            path[i] = (cv2.resize(a_chan, size, cv2.INTER_MAX)>0).astype(float)
        path = np.flip(path, axis=1)
        base = (np.sum(path, axis=0)>0).astype(float)
        B[j] = base
    return B

def pic_to_values(pic):
    X, Y = 0, 0
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            if pic[y,x]:
                X += pic[y,x]*x
                Y += pic[y,x]*y
    summed = pic.sum()
    X /= pic.shape[0]*summed
    Y /= pic.shape[0]*summed
    R = np.sqrt(summed)/pic.shape[0]
    return X.item(), Y.item(), R.item()

def format(X, size=(32,32)):
    x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for i, scene in enumerate(X):
        channels = [(scene==j).astype(float) for j in range(1,8)]
        x[i] = np.stack([(cv2.resize(c, size, cv2.INTER_MAX)>0).astype(float) for c in channels])
    return x

if __name__ == "__main__":
    model = FlowNet(7, 16)
    model2 = FlowNet(8, 16)
    model.load_state_dict(T.load("saves/imaginet-c16-all.pt"))
    model2.load_state_dict(T.load("saves/imaginet2-c16-all.pt"))
    solve(model, model2)
