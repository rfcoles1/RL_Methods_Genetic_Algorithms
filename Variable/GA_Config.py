import os
import sys
import gym
import time

sys.path.insert(0,'../../')
import Var_Games

class Config:
    env_name = 'Var_Acro-v0' 
    game = gym.make(env_name)

    s_size = game.observation_space.shape[0] 
    a_size = game.action_space.n

    num_policies = 100 #size of population
    num_generations = 25 #number of generations
    checkpoint_freq = 1 #number of generations between checkpoints
    num_iterations = 25 #number of games each policy is evaluated for, score is averaged
    
    #define the network 
    num_layers = 2 
    num_hidden = 128
    lr = 0.05
    mutate_per = 0.75 
    sigma = 0.1

    model_path = './models/' + str(env_name) + '/' #+ '_' + str(time.ctime()) + '/' 
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    load_model = False
    version_to_load = '0.npz'
    #define when the game is solved,
    #must have an average score higher than the benchmark over a series of consecutive games
    
    if env_name == 'Var_MC-v0':
        score_to_increase = -125
        h_step = 0.1
        start_h = 0
        final_h = 1
        score_to_solve = -110
        episodes_to_solve = 5
    if env_name == 'Var_Acro-v0':
        score_to_increase = -100
        h_step = 0.2
        start_h = -1
        final_h = 1
        score_to_solve = -50
        episodes_to_solve = 5
    if env_name == 'Var_Carnot-v0':
        score_to_increase = 0.36 
        h_step = -1e-5
        start_h = 20e-5
        final_h = 1e-5
        score_to_solve = 0.39
        episodes_to_solve = 1

    
