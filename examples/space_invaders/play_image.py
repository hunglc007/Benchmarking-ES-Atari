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


class InvadersModelMihn(nn.Module):
    def __init__(self, num_features):
       super(InvadersModelMihn, self).__init__()
       self.main = nn.Sequential(
       nn.Conv2d(3, num_features, 8, 4, 1, bias=False),
       nn.LeakyReLU(0.2, inplace=True),
       nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
       nn.BatchNorm2d(num_features * 2),
       nn.LeakyReLU(0.2, inplace=True))

       self.fc1 = nn.Linear(32*15*15, 256)
       self.fc2 = nn.Linear(256, 6)
    def forward(self, input):
        main = self.main(input)
        main = main.view(-1, 32*15*15)
        main = F.relu(self.fc1(main))
        main = self.fc2(main)
        return main
# model = InvadersModel(num_features)
model = InvadersModelMihn(num_features)
if cuda:
    model = model.cuda()

def get_reward(weights, model, render=False):
  cloned_model = copy.deepcopy(model)
  for i, param in enumerate(cloned_model.parameters()):
    try:
      param.data = weights[i]
    except:
      param.data = weights[i].data
  env = gym.make("SpaceInvaders-v0")
  ob = env.reset()
  done = False
  total_reward = 0
  while not done:
    if render:
      env.render()
      time.sleep(0.05)
    image = transform(Image.fromarray(ob))
    image = image.unsqueeze(0)
    if cuda:
      image = image.cuda()
    prediction = cloned_model(Variable(image))
    action = np.argmax(prediction.cpu().data.numpy())
    ob, reward, done, _ = env.step(action)
    total_reward += reward
  env.close()
  # print(total_reward)
  return total_reward


weights = pickle.load(open(os.path.abspath(args.weights_path), 'rb'))

final_reward = get_reward(weights, model, render=True)
print(final_reward)