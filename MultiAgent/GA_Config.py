import os
import sys
import gym
import time

sys.path.insert(0,'../')
import MultiAgent_Games

class Config:

    env_name = 'Bees-v1'
    mode = 'discrete' #either discrete or continuous depending on the action set of the game

    game = gym.make(env_name)
    
    max_episode_steps = game._max_episode_steps
    #s_size = game.observation_space.shape[0] 
    s_size = len(game.reset().flatten())
    
    if mode == 'discrete':
        a_size = game.action_space.n
    else: #continuous action set
        a_size = game.action_space.shape[0]
        a_bounds = [game.action_space.low, game.action_space.high]
        a_range = game.action_space.high - game.action_space.low

    num_policies = 100 #size of population
    num_generations = 100000 #number of generations
    checkpoint_freq = 10 #number of generations between checkpoints
    num_iterations = 10 #number of games each policy is evaluated for, score is averaged
    
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
    score_to_solve = 10
    episodes_to_solve = 10

    load_model = False
    version_to_load = '0.npz'

