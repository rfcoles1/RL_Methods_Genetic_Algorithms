import os 
import sys
import numpy as np
import gym
import time

from GA_Config import Config
from GA_Network import Network
from GA_Helper import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('n', type = int)
args = parser.parse_args()
#f = open("_results" + str(args.n) + ".txt", 'a')

config = Config()
env = gym.make(Config.env_name)


#fill the population
population = []
for i in range(config.num_policies):
    pop = Network(config)
    population.append(pop)

if config.load_model == True:
    weights = np.load(config.model_path + config.version_to_load)
    population[0].load_net(weights['w_in'], weights['w_h'], weights['w_out'])

#every episode evaluates each policy
for episode in range(config.num_generations):
    start = time.time()
    Reward = np.zeros(config.num_policies)
    for policy in range(config.num_policies):
        curr_pol = population[policy]
        for i in range(config.num_iterations):
            Reward[policy] += curr_pol.playthrough(env)

    Reward /= config.num_iterations
    print('Episode: %i, Mean: %.2f, Max: %.2f, Time: %.2f' %(episode, np.mean(Reward), np.max(Reward), time.time() - start))
    #np.savetxt(f, np.reshape(Reward, [1,len(Reward)]))

    #sort the policies by score achieved and remove the lowest scoring 
    l1, l2 = zip(*sorted(zip(Reward, population), key = lambda x: x[0]))
    population = list(l2[int(config.mutate_per*config.num_policies):])
    Reward = list(l1[int(config.mutate_per*config.num_policies):])

    #save and check if solved
    if (episode % config.checkpoint_freq == 0) and (episode != 0):
        network = population[-1] #take the best network

        summed_reward = 0
        for i in range(config.episodes_to_solve):
            reward = network.playthrough(env)
            summed_reward += reward


        score = summed_reward/config.episodes_to_solve
        print('Average score over ' + \
            str(config.episodes_to_solve) + ' episodes: ' + str(score))
        np.savez(config.model_path + str(episode) + '.npz',\
            w_in = network.w_in, w_h = network.w_hidden, w_out = network.w_out)
        if (score > config.score_to_solve):
            print 'The game is solved!'
            break      
        
        sys.stdout.flush()

    #refill the population      
    mutants = []
    for i in range(int(config.mutate_per*config.num_policies)):
        curr_pol = np.random.choice(population) #p = Reward/sum(Reward))
        new_pol = copy_net(curr_pol)
        mutation(new_pol)
        mutants.append(new_pol)
    population += mutants

#f.close()    
        
