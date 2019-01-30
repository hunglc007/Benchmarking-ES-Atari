import argparse
from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import torch.nn.functional as F
import random
import sys
import time

# from evostra import EvolutionStrategy
from strategies.evolution import EvolutionModule
import gym
from gym import logger as gym_logger
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from torchvision import transforms

gym_logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights_path', type=str, default='model_space_invaders_image.p', help='Path to save final weights')
parser.add_argument('-c', '--cuda', action='store_true', help='Whether or not to use CUDA')
parser.set_defaults(cuda=False)

args = parser.parse_args()

cuda = args.cuda and torch.cuda.is_available()

num_features = 16
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

model = nn.Sequential(
    nn.Linear(128, 200),
    nn.ReLU(),
    nn.Linear(200, 500),
    nn.ReLU(),
    nn.Linear(500, 6),
    nn.Softmax(1)
)


if cuda:
    model = model.cuda()

env = gym.make("SpaceInvaders-ram-v0")

def get_reward(weights, model, render=False):
    global env

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data

    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.005)
        batch = torch.from_numpy(ob[np.newaxis,...]).float()
        if cuda:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = cloned_model(Variable(batch))
        action = prediction.data.cpu().numpy().argmax()
        ob, reward, done, _ = env.step(action)

        total_reward += reward
    env.close()
    return total_reward

weights = pickle.load(open(os.path.abspath(args.weights_path), 'rb'))

final_reward = get_reward(weights, model, render=True)
print(final_reward)