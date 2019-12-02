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

#fill the population
population = []
for i in range(config.num_policies):
    pop = Network(config)
    population.append(pop)

#every episode evaluates each policy
for episode in range(config.num_generations):
    start = time.time()
    Reward = np.zeros(config.num_policies)
    for policy in range(config.num_policies):
        curr_pol = population[policy]
        for i in range(config.num_iterations):
            env.reset()
            total_reward = 0           
            eps = 0
            while True:
                s0 = env.get_state(0)
                a0 = curr_pol.predict(s0.flatten())
                a0 = np.argmax(a0)

                _, reward, done, _ = env.step(a0, 0)
                
                s1 = env.get_state(1)
                a1 = curr_pol.predict(s1.flatten())
                a1 = np.argmax(a1)
                _, reward, done , _ = env.step(a1, 1)
                
                total_reward = reward
                eps += 1
                if done:
                    Reward[policy] += total_reward
                    break 

    Reward /= config.num_iterations
    print('Episode: %i, Mean: %.2f, Max: %.2f, Time: %.2f' %(episode, np.mean(Reward), np.max(Reward), time.time() - start))

    #sort the policies by score achieved and remove the lowest scoring 
    l1, l2 = zip(*sorted(zip(Reward, population), key = lambda x: x[0]))
    population = list(l2[int(config.mutate_per*config.num_policies):])
    Reward = list(l1[int(config.mutate_per*config.num_policies):])
    
    #save and check if solved
    if (episode % config.checkpoint_freq == 0) and (episode != 0):
        network = population[-1] #take the best network

        summed_reward = 0
        for i in range(config.episodes_to_solve):
            eps_reward = 0
            env.reset()
            total_reward = 0           
            eps = 0
            while True:
                s0 = env.get_state(0)
                a0 = network.predict(s0.flatten())
                a0 = np.argmax(a0)

                _, reward, done, _ = env.step(a0, 0)
                
                #s1 = env.get_state(1)
                #a1 = curr_pol.predict(s1.flatten())
                #a1 = np.argmax(a1)
                #_, reward, done , _ = env.step(a1, 1)
                
                total_reward = reward
                eps += 1
                if done:
                    break 

            summed_reward += total_reward

        score = float(summed_reward)/config.episodes_to_solve
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
        
