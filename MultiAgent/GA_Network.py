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

    def playthrough1(self, env, netID):
        s = env.reset()
        total_reward = 0

        while True:
            a = self.predict(s.flatten())
            a = np.argmax(a)
            s, reward, done, _ = env.step(a, netID)

            total_reward = reward
            if done:
                return total_reward
    
    def playthrough2(self, env, netID):
        s = env.reset()
        total_reward = 0

        while True:
            s0 = env.get_state(0)
            a0 = self.predict(s0.flatten())
            a0 = np.argmax(a0)
            _, reward, done, _ = env.step(a0, 0)

            s1 = env.get_state(1)
            a1 = self.predict(s1.flatten())
            a1 = np.argmax(a1)
            _, reward, done, _ = env.step(a1, 1)

            total_reward = reward
            if done:
                return total_reward
                
    def load_net(self, w_i, w_h, w_o):
        self.w_in = w_i
        self.w_hidden = w_h
        self.w_out = w_o



