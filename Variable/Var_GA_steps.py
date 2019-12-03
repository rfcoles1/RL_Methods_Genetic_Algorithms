import os 
import sys
import numpy as np
import gym
import time

from GA_Config import Config
from GA_Network import Network
from GA_Helper import *

config = Config()
env = gym.make(Config.env_name).unwrapped

f = open("Results.txt", 'w')

#fill the population
population = []
for i in range(config.num_policies):
    pop = Network(config)
    population.append(pop)

if config.load_model == True:
    weights = np.load(config.model_path + config.version_to_load)
    population[0].load_net(weights['w_in'], weights['w_h'], weights['w_out'])

#every episode evaluates each policy
h = config.start_h

for episode in range(config.num_generations):
    start = time.time()
    Reward = np.zeros(config.num_policies)
    for policy in range(config.num_policies):
        curr_pol = population[policy]
        for i in range(config.num_iterations):
            Reward[policy] += curr_pol.playthrough(env, h)
    Reward /= config.num_iterations
    print("Episode: %i, h_param: %.5f, Mean: %.2f, Max: %.2f, Time: %.2f" %(episode, h, np.mean(Reward), np.max(Reward), time.time() - start))

    tofile = np.insert(Reward, 0, h)
    np.savetxt(f, np.reshape(tofile, [1, len(tofile)]))

    #sort the policies by score achieved and remove the lowest scoring 
    l1, l2 = zip(*sorted(zip(Reward, population), key = lambda x: x[0]))
    population = list(l2[int(config.mutate_per*config.num_policies):])
    Reward = list(l1[int(config.mutate_per*config.num_policies):])

    score = Reward[-1]
    if np.round(h,5) < config.final_h and score > config.score_to_increase:
        print("Adjusting h")
        h += config.h_step
    elif np.round(h,5) == config.final_h and score > config.score_to_solve:
        print('The game is solved!')
        break      
        

    """
    #save and check if solved
    if (episode % config.checkpoint_freq == 0) and (episode != 0):
        network = population[-1] #take the best network

        summed_reward = 0
        for i in range(config.episodes_to_solve):
            reward = network.playthrough(env, h)
            summed_reward += reward


        score = summed_reward/config.episodes_to_solve
        print('Average score over ' + \
            str(config.episodes_to_solve) + ' episodes: ' + str(score))
        np.savez(config.model_path + str(episode) + '.npz',\
            w_in = network.w_in, w_h = network.w_hidden, w_out = network.w_out, h = h)
        
        #SOMETIMES NEEDS TO BE GREATER/LESS
                sys.stdout.flush()
    """
    #refill the population      
    mutants = []
    for i in range(int(config.mutate_per*config.num_policies)):
        curr_pol = np.random.choice(population) #p = Reward/sum(Reward))
        new_pol = copy_net(curr_pol)
        mutation(new_pol)
        mutants.append(new_pol)
    population += mutants

    
        
