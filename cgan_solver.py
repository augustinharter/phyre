#%%
from action_cgan import *
from phyre_utils import pic_to_action_vector, pic_hist_to_action
from phyre_rolllout_collector import collect_interactions
import torch as T
import phyre
import numpy as np
import cv2
import json
import itertools
from matplotlib import pyplot as plt
import os

#%%
def solve(tasks, generator, save_images=False, force_collect=False, static=256, show=False):
    # Collect Interaction Data
    data_path = './data/cgan_solver'
    if not os.path.exists(data_path+'/interactions.pickle') or force_collect:
        os.makedirs(data_path, exist_ok=True)
        wid = generator.width
        print("Collecting Data")
        collect_interactions(data_path, tasks, 10, stride=1, size=(wid,wid), static=static)
    with open(data_path+'/interactions.pickle', 'rb') as fs:
        X = T.tensor(pickle.load(fs), dtype=T.float)
    with open(data_path+'/info.pickle', 'rb') as fs:
        info = pickle.load(fs)
        tasklist = info['tasks']
        positions = info['pos']
        orig_actions = info['action']
    print('loaded dataset with shape:', X.shape)
    #data_set = T.utils.data.TensorDataset(X)
    #data_loader = T.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=False)

    # Sim SETUP
    print('Succesfull collection for tasks:\n', tasklist)
    eval_setup = 'ball_within_template'
    sim = phyre.initialize_simulator(tasklist, 'ball')
    eva = phyre.Evaluator(tasklist)

    # Solve Loop
    error = np.zeros((X.shape[0],3))
    generator.eval()
    solved, tried = 0, 0
    for i,task in enumerate(tasklist):
        # generate 'fake'
        noise = T.randn(1, generator.noise_dim)
        with T.no_grad():
            fake = generator((X[i,:generator.s_chan])[None], noise)[0,0]
        #action = np.array(pic_to_action_vector(fake, r_fac=1.8))
        action = np.array(pic_to_action_vector(fake.numpy(), r_fac=1))
        raw_action = action.copy()
        
        # PROCESS ACTION
        print(action, 'raw')
        # shift by half to get relative position
        action[:2] -= 0.5
        # multiply by half because extracted scope is already half of the scene
        action[:2] *= 0.5
        # multiply by 4 because action value is always 4*diameter -> 8*radius, but scope is already halfed -> 8*0.5*radius
        action[2] *=4
        # finetuning
        action[2] *= 1.0
        print(action, 'relativ')
        pos = positions[i]
        print(pos)
        action[:2] += pos
        print(action, 'added')
        res = sim.simulate_action(i, action, need_featurized_objects=True)
        
        # Noisy tries while invalid actions
        t = 0
        temp = 1
        base_action = action
        while res.status.is_invalid() and t <200:
            t += 1
            action = base_action + (np.random.rand(3)-0.5)*0.01*temp
            res = sim.simulate_action(i, action,  need_featurized_objects=False)
            temp *=1.01
        print(action, 'final action')

        # Check for and log Solves
        if not res.status.is_invalid():
            tried += 1
        if res.status.is_solved():
            solved +=1
        print(orig_actions[i], 'orig action')
        print(task, "solved", res.status.is_solved())
        error[i] = orig_actions[i]-base_action

        # Visualization
        if show:
            x, y, d = np.round(raw_action*fake.shape[0])
            y = fake.shape[0]-y
            print(x,y,d)

            def generate_crosses(points):
                xx = []
                yy = []
                for x,y in points:
                    xx.extend([x,x+1,x-1,x,x])
                    yy.extend([y,y,y,y+1,y-1])
                return xx, yy

            xx, yy = [x,(x+d) if (x+d)<fake.shape[0]-1 else 62,x-d,x,x], [y,y,y, (y+d) if (y+d)<fake.shape[0]-1 else 62,y-d]
            xx, yy = generate_crosses(zip(xx,yy))
            fake[yy,xx] = 0.5
            os.makedirs(f'result/cgan_solver/vector_extractions',exist_ok=True)
            plt.imsave(f'result/cgan_solver/vector_extractions/{i}.png',fake)
            if not res.status.is_invalid():
                os.makedirs(f'result/cgan_solver/scenes',exist_ok=True)
                plt.imsave(f'result/cgan_solver/scenes/{i}.png',res.images[0,::-1])
            else:
                print("invalid")
                plt.imshow(phyre.observations_to_float_rgb(sim.initial_scenes[i]))
                plt.show()

    print("solving percentage:", solved/tried, 'overall:', tried)
    print("mean x error:", np.mean(error[:,0]), 'mean x abs error:', np.mean(np.abs(error[:,0])))
    print("mean y error:", np.mean(error[:,1]), 'mean y abs error:', np.mean(np.abs(error[:,1])))
    print("mean r error:", np.mean(error[:,2]), 'mean r abs error:', np.mean(np.abs(error[:,2])))

class CganInteractionSolver():
    """
    USAGE:
    solver = CganInteractionSolver("path/to/model")
    solver.solve_interactions(values_batch, scenes_batch)
    """

    def __init__(self, model_path:str = "./saves/action_cgan/3conv64-128/generator.pt", width:int = 64):
        self.width = width

        # Loading state_dict
        state_dict = T.load(model_path, map_location=T.device('cpu'))

        # Extracting Model parameters from state_dict
        in_channels = state_dict['encoder.0.weight'].shape[1]
        layer_numbers = set(int(key[11:13].strip('.')) for key in state_dict if key.startswith('conv_model'))
        last_layer = max(layer_numbers)
        out_channels = state_dict[f'conv_model.{last_layer}.weight'].shape[1]
        n_encoder_layers = len(set(int(key[8:10].strip('.')) for key in state_dict if key.startswith('encoder')))
        print(f'Loaded model with: width {width}, in_chs {in_channels}, out_chs {out_channels}, folds {n_encoder_layers-1}')

        # Loading model:
        self.generator = Generator(width, 100, in_channels, out_channels, folds=n_encoder_layers-1)
        self.generator.load_state_dict(state_dict)
        self.generator.eval()

    def solve_interactions(self, values_batch, scenes_batch, same_noise=False, show=False):
        """
        Inputs:

        [green_ball_radius
        green_ball_x_t-1, 
        green_ball_y_t-1, 
        green_ball_x_t, 
        green_ball_y_t, 
        green_ball_x_t+1, 
        green_ball_y_t+1]  # all values are from 0 to 1, radius from 0 to 0.25
        
        [7-channels scene picture]  # or maybe 6-channels without red ball, doesnt matter if action ball channel is empty


        Returns:
        [x, y, r]  # red ball action

        [confidence value]  # continues confidence value between 0 and 1 that indicated the probability that the output action (red ball) will lead to the specified in the input values t+1 green-ball values [green_ball_x_t+1, green_ball_y_t+1].
        0 - [x, y, r] action will not lead to [green_ball_x_t+1, green_ball_y_t+1]
        0.5 - 50% chance that the action [x, y, r] will  lead to [green_ball_x_t+1, green_ball_y_t+1]
        1.0 - 100% chance that the action [x, y, r] will  lead to [green_ball_x_t+1, green_ball_y_t+1]
        """

        values_batch = np.array(values_batch)
        print("values shape:", values_batch.shape, "scenes shape:", scenes_batch.shape)
        channels = T.zeros(len(values_batch), 4, self.width, self.width)
        for i in range(len(values_batch)):
            values = values_batch[i]
            r = values[0] # r € [0,0.25]  scope is already half so orig_r*2 = scope_r = orig_diameter
            coords = np.array(values[1:])
            rel_xcoords = coords[[0,2,4]]-coords[2]
            rel_ycoords = coords[[1,3,5]]-coords[3]
            rel_normed_xcoords = (rel_xcoords /0.5)+0.5 # Scope is half of original (factor 1/2) 
            rel_normed_ycoords = (rel_ycoords /0.5)+0.5 # and shifted because center is 0.5
            xminus, x, xplus = rel_normed_xcoords
            yminus, y, yplus = rel_normed_ycoords

            # Pad, center, zoom and flip scene channel:
            scene = np.max(scenes_batch[i,:], axis=0)
            width = 128
            wh = width//2
            startx = int(256*(coords[2]))
            starty = int(256*(1-coords[3]))
            padded_scene = np.pad(scene, ((wh,wh), (wh,wh)))
            centered_scene = padded_scene[starty:starty+width, startx:startx+width]
            zoomed_scene = cv2.resize(centered_scene, (64,64))
            flipped_scene = np.flip(zoomed_scene, axis=0)

            channels[i,0] = self.draw_ball(xminus, yminus, r)
            channels[i,1] = self.draw_ball(x, y, r)
            channels[i,2] = self.draw_ball(xplus, yplus, r)
            channels[i,3] = T.tensor(flipped_scene.copy())
        
        # Generating Predictions
        noise = T.randn(len(values_batch), self.generator.noise_dim)
        if same_noise:
            noise = T.randn(self.generator.noise_dim).repeat((len(values_batch),1)) 
            # Noise makes a big difference, this is the same noise for the whole batch 
        with T.no_grad():
            predictions = self.generator(channels, noise)
        if show:
            images = (T.sum(T.cat((channels, predictions), dim=1), dim=1)>0.01).float()
            plt.imshow(T.sum(channels, dim=1)[0])
            plt.show()
            plt.imshow(predictions[0,0])
            plt.show()
            plt.imshow(images[0])
            plt.show()

        actions = []
        confidence = []
        for i, pic in enumerate(predictions):
            # PROCESS ACTION
            action = np.array(pic_to_action_vector(pic[0]))
            # shift by half to get relative position
            action[:2] -= 0.5
            # multiply by half because extracted scope is already half of the scene
            action[:2] *= 0.5
            # multiply by 4 because action value is always 4*diameter -> 8*radius, but scope is already halfed -> 8*0.5*radius
            action[2] *= 4
            # finetuning
            action[2] *= 1.0
            pos = np.array(values_batch[i][3:5])
            action[:2] += pos

            actions.append(action)
            confidence.append(1) # TODO Calculate real Confidence Value
        
        return actions, confidence
    
    def draw_ball(self, x, y, r):
        x = int(self.width*x)
        y = int(self.width*(1-y))
        r = self.width*r
        X = T.arange(self.width).repeat((self.width, 1)).float()
        Y = T.arange(self.width).repeat((self.width, 1)).transpose(0, 1).float()
        X -= x # X Distance
        Y -= y # Y Distance
        dist = (X.pow(2)+Y.pow(2)).pow(0.5)
        return (dist<r).float()
            
        

#%%
if __name__ == "__main__":
    # USAGE:
    # solver = CganInteractionSolver()
    # solver.solve_interactions(values_batch, scenes_batch)

    # TESTING
    # Setup
    solver = CganInteractionSolver("./saves/action_cgan/3conv64-128/generator.pt", width = 64)
    data_path = './data/cgan_solver'
    if not os.path.exists(data_path+'/interactions.pickle'):
        os.makedirs(data_path, exist_ok=True)
        wid = generator.width
        print("Collecting Data")
        collect_interactions(data_path, tasks, 10, stride=1, size=(wid,wid), static=static)
    with open(data_path+'/interactions.pickle', 'rb') as fs:
        X = T.tensor(pickle.load(fs), dtype=T.float)
    with open(data_path+'/info.pickle', 'rb') as fs:
        info = pickle.load(fs)
        tasklist = info['tasks']
        positions = info['pos']
        orig_actions = info['action']
    print('loaded dataset with shape:', X.shape)
    data_set = T.utils.data.TensorDataset(X)
    data_loader = T.utils.data.DataLoader(data_set, batch_size=4, shuffle=True)

    # Testing Loop
    for i, (X,) in enumerate(data_loader):
        scenes_batch = np.flip(np.array([np.pad(cv2.resize(scene.numpy(), (128,128)), ((64,64), (64,64))) for scene in X[:,3]]), axis = 1)[:,None]
        print('real sum', T.sum(X[0,0]))
        minus = pic_to_action_vector(X[0,0])
        zero = pic_to_action_vector(X[0,1])
        plus = pic_to_action_vector(X[0,2])
        print('drawn sum', T.sum(solver.draw_ball(*minus)))
        print(minus, zero, plus)
        # method expects diameter in full scene == radius in half scene
        values_batch = np.array([[zero[2], minus[0], minus[1], zero[0], zero[1], plus[0], plus[1]]])
        values_batch[1:] *= 0.5
        plt.imshow(solver.draw_ball(*pic_to_action_vector(X[0,2])))
        plt.show()
        plt.imshow(X[0,2])
        plt.show()
        plt.imshow(X[0,2]-solver.draw_ball(*pic_to_action_vector(X[0,2])))
        plt.show()
        # solving
        print(solver.solve_interactions(values_batch, scenes_batch, show = True))

    wid = 64
    generator = Generator(wid, 100, 4, 1, folds=3)
    generator.load_state_dict(T.load("./saves/action_cgan/3conv64-128/generator.pt", map_location=T.device('cpu')))

    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    all_tasks = train_ids+dev_ids+test_ids
    template13_tasks = [t for t in all_tasks if t.startswith('00013:')]
    template2_tasks = [t for t in all_tasks if t.startswith('00002:')]

    solve(template2_tasks, generator, force_collect=False, static=128, show=False)
