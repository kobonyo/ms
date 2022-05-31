import argparse
import random
import os
import math
from os import path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from MSDataLoader import CustomDataset
from MS import MSGame as game
from torch.utils.data import DataLoader
from NNet import MSNet
from TrainModel import Model
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--root_path', help = "root dir", type = str, default = "./ms/")
parser.add_argument('-f', '--model_file', type = str, help = "path to value network", default = "./ms/model.ckpt")
parser.add_argument('-n', '--train_results', type = str, help = "train log file name", default = "./ms/resTrain.txt")
parser.add_argument('-t', '--eval_results', type = str, help = "eval log file name", default = "./ms/resEval.txt")
parser.add_argument('-e', '--episodes', type = int, help = "number of episodes to play", default = 2000)
parser.add_argument('-b', '--batch_size', type = int, help = "batch size", default = 64)
parser.add_argument('-x', '--mx_epochs', type = int, help = "Max epochs to play", default = 50)
parser.add_argument('-w', '--num_workers', type = int, help = "number of workers", default = 8)
parser.add_argument('-a', '--replay_size', type = int, help = "replay memeory size", default = 50000)
parser.add_argument('-s', '--test_size', type = float, help = "test size per cent", default = 0.25)
parser.add_argument('-k', '--sample_train', type = int, help = "replay sample size to train on", default = 10000)
parser.add_argument('-l', '--batch_iterations', type = int, help = "batch size for the playout", default = 20)
parser.add_argument('-o', '--optimizer', type = str, help = "optimizer", default = "adam")
parser.add_argument('-c', '--loss', type = str, help = "loss", default = "mse")
parser.add_argument('-g', '--lr', type = float, help = "learnig rate", default = 0.0001)
parser.add_argument('-v', '--print_every', type = int, help = "freq of  printing the logs", default = 6)
parser.add_argument('-p', '--pos_reshape', type = int, help = "positive threshold", default = 60)
parser.add_argument('-d', '--past_states', type = int, help = "past state to look at", default = 6)

args = parser.parse_args()


GLOBAL_SCORE = 0
TRAIN = deque([], maxlen = args.replay_size)
PAST_STATES =  deque([], maxlen = args.past_states)
EPS = 0.01


model = Model(MSNet(), 
              args.mx_epochs, 
              args.optimizer,
              args.loss,
              args.model_file,
              args.batch_size,
              args.lr,
              29,
              0
              )

def e_greedy():
    global EPS
    p = np.random.random()
    if p < EPS: return True
    return False

def do_undo_move(state, action):
    state.make_move(action)
    state_data = state.current_state()
    state.undo_move(action)
    return state_data

def look_ahead(state):
    action_value, action_data  = [] , []
    actions  = state.actions()
    for a in state.actions():
        state_data = do_undo_move(state, a)
        input_tensor = torch.tensor(state_data).view(1, 1, 30, 30)
        action_value.append(model.predict(input_tensor))
        action_data.append(state_data)
    action_value = torch.tensor(action_value).view(-1, 1)  
    action_ix = torch.argmax(action_value).item()
    state.data.append(action_data[action_ix])
    assert(len(actions) == len(state.actions()))
    print(action_value)
    return actions[action_ix]

def value(state, eval = False):
    actions = state.actions()
    if eval : return look_ahead(state)
    if e_greedy():
        action = random.choice(state.actions())  
        state.data.append(do_undo_move(state, action))
        assert(len(actions) == len(state.actions()))
        return action
    else:return look_ahead(state)
        

def sample_replay(data, sz):
    if len(data) >= sz: data = random.sample(data, sz)    
    X , y = [], []
    for d in data:
        X.append(d[0])
        y.append(d[1])        
    X = torch.cat(X).view(-1, 1, 30, 30)
    y = torch.tensor(y).view(-1, 1).float()
    assert(X.size(0) == y.size(0))
    return X , y

def train_model():
    X , y = sample_replay(TRAIN, args.sample_train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size = args.batch_size, num_workers = args.num_workers)
    valid_loader = DataLoader(CustomDataset(X_test, y_test), batch_size = args.batch_size, num_workers = args.num_workers)
    model.trainer(train_loader, valid_loader)  

def log_summaries(state, log_path):
    global GLOBAL_SCORE
    if state.reward() > GLOBAL_SCORE:
        GLOBAL_SCORE = state.reward()
        print(f"current score {state.reward()}, global score {GLOBAL_SCORE}")
        state.write_results(log_path)

def label_states(state, log_path):
    global TRAIN
    total_reward = state.reward()
    step_reward = 1.0
    assert(len(state.data) == len(state.reward_list))
    for d, r in zip(state.data, state.reward_list):
        if len(TRAIN) > args.replay_size:TRAIN.popleft()
        TRAIN.append([torch.tensor(d).view(1, 1, 30, 30), (step_reward+total_reward+r)/121.0])
    log_summaries(state, log_path)

def ms_batch_playout():
    print(f"Playing ...")
    for _ in range(args.batch_iterations):
        state = game()
        while state.available_moves():
            best_action = value(state)
            state.make_move(best_action, True)
        assert(len(state.reward_list) == len(state.data) ==  len(state.episode_moves))
        label_states(state, args.train_results)

def evaluate():
    print(f"Evaluating ...")
    for _ in range(args.batch_iterations):
        state = game()
        while state.available_moves():
            best_action = value(state, True)
            state.make_move(best_action, True)
        label_states(state, args.eval_results)
        assert(len(state.reward_list) == len(state.data) ==  len(state.episode_moves))
    
if __name__ == "__main__":
    if not path.exists(args.root_path):os.mkdir(args.root_path)
    for iteration in range(args.episodes):
        ms_batch_playout()
        train_model()
        evaluate()
    print(f"Done! ...")