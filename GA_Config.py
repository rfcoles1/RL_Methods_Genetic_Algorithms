import os
import sys
import gym
import time

sys.path.insert(0,'../')
import Games

class Config:

    env_name = 'CartPole-v0' 
    mode = 'discrete' #either discrete or continuous depending on the action set of the game

    if env_name == 'CartPole-v0' or env_name == 'MountainCar-v0' or env_name == 'Acrobot-v1':
        mode = 'discrete'
    elif env_name == 'MountainCarContinuous-v0' or env_name == 'Pendulum-v0': 
        mode = 'continuous'

    game = gym.make(env_name)

    s_size = len(game.reset()) 
    if mode == 'discrete':
        a_size = game.action_space.n
    else: #continuous action set
        a_size = game.action_space.shape[0]
        a_bounds = [game.action_space.low, game.action_space.high]
        a_range = game.action_space.high - game.action_space.low

    num_policies = 100 #size of population
    num_generations = 100 #number of generations
    checkpoint_freq = 1 #number of generations between checkpoints
    num_iterations = 3 #number of games each policy is evaluated for, score is averaged
    
    #define the network 
    num_layers = 2 
    num_hidden = 128
    lr = 0.05
    mutate_per = 0.75 
    sigma = 0.1

    model_path = './models/' + str(env_name) + '/' #+ '_' + str(time.ctime()) + '/' 
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #define when the game is solved,
    #must have an average score higher than the benchmark over a series of consecutive games
    score_to_solve = 195
    episodes_to_solve = 100

    load_model = False
    version_to_load = '0.npz'
