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

config = Config()
env = gym.make(Config.env_name).unwrapped

#fill the population
population = []
for i in range(config.num_policies):
    pop = Network(config)
    population.append(pop)

if config.load_model == True:
    weights = np.load(config.model_path + config.version_to_load)
    population[0].load_net(weights['w_in'], weights['w_h'], weights['w_out'])

#every episode evaluates each policy

for switch in [1, 2, 3, 4, 5]:
    f = open("Results_Acro_switch" + str(switch) + "_" + str(args.n) + ".txt", 'w')
    h = 0
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

        if episode >= switch-1:
            h = 1

        #refill the population      
        mutants = []
        for i in range(int(config.mutate_per*config.num_policies)):
            curr_pol = np.random.choice(population) #p = Reward/sum(Reward))
            new_pol = copy_net(curr_pol)
            mutation(new_pol)
            mutants.append(new_pol)
        population += mutants
    
    f.close()
        
