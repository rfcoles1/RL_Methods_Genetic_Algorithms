import numpy as np 
import time


class Network():
    def __init__(self, config):
      
        self.config = config

        #input layer
        self.w_in = np.random.randn(config.num_hidden,config.s_size)/np.sqrt(config.s_size)
        self.num_in = len(self.w_in.flatten())

        #hidden layers, number is defined in config class
        self.w_hidden = [] #stores the weights for each layer
        self.num_weights = [] #stores how many weights are in each layer
        for _ in range(config.num_layers):
            h = np.random.randn(config.num_hidden,config.num_hidden)/np.sqrt(config.num_hidden)
            num_h = len(h.flatten())
            self.w_hidden.append(h)
            self.num_weights.append(num_h)

        #output layer 
        self.w_out = np.random.randn(config.a_size,config.num_hidden)/np.sqrt(config.num_hidden)      
        self.num_out = len(self.w_out.flatten())
       
        #total number of weights
        self.total_num = self.num_in + self.num_out + int(np.sum(self.num_weights))

    def predict(self, s):
        h = np.dot(self.w_in,s) #input to hidden layer
        h[h<0] = 0 #relu
        for i in range(len(self.w_hidden)):
            hnew = np.dot(self.w_hidden[i],h)
            hnew[hnew<0] = 0
            h = hnew
        out = np.dot(self.w_out,h) #hidden layer to output
        out = 1.0/(1.0 + np.exp(-out)) #sigmoid
        return out

    def playthrough(self, env, h):
        s = env.reset(h)
        total_reward = 0
        eps = 0
        max_eps = 200
        reward_history = np.zeros(max_eps) 
        
        while eps < 200:
            #perform action based on this policy
            a = self.predict(s)
            a = np.argmax(a)
            s, reward, done, _ = env.step(a)

            if self.config.env_name == 'Var_Carnot-v0':
                reward_history[eps] = reward
                eps += 1
                if done: 
                    return max(reward_history)
            else:
                '''when the score is summed through entire game'''
                total_reward += reward
                '''when only final score is taken'''
                #total_reward = reward
                eps += 1
                if done: 
                    return total_reward

        if self.config.env_name == 'Var_Carnot-v0':
            return max(reward_history)
        else:
            return total_reward 
            
    def load_net(self, w_i, w_h, w_o):
        self.w_in = w_i
        self.w_hidden = w_h
        self.w_out = w_o



