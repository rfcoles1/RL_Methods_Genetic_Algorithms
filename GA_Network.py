import numpy as np 

class Network():
    def __init__(self, config):
      
        self.config = config

        #input layer
        self.w_in = np.random.randn(config.num_hidden,config.s_size)/np.sqrt(config.s_size)
        self.num_in = len(self.w_in.flatten())

        #hidden layers, number is defined in config class
        self.weights = [] #stores the weights for each layer
        self.num_weights = [] #stores how many weights are in each layer
        for _ in range(config.num_layers):
            h = np.random.randn(config.num_hidden,config.num_hidden)/np.sqrt(config.num_hidden)
            num_h = len(h.flatten())
            self.weights.append(h)
            self.num_weights.append(num_h)

        #output layer 
        self.w_out = np.random.randn(config.a_size,config.num_hidden)/np.sqrt(config.num_hidden)      
        self.num_out = len(self.w_out.flatten())
       
        #total number of weights
        self.total_num = self.num_in + self.num_out + int(np.sum(self.num_weights))

    def predict(self, s, w_in, w_out, h_weights):
        h = np.dot(w_in,s) #input to hidden layer
        h[h<0] = 0 #relu
        
        for i in range(len(h_weights)):
            hnew = np.dot(self.weights[i],h)
            hnew[hnew<0] = 0
            h = hnew

        out = np.dot(w_out,h) #hidden layer to output
        out = 1.0/(1.0 + np.exp(-out)) #sigmoid
        return out

    def playthrough(self, env, w_in, w_out, h_weights):
        s = env.reset()
        total_reward = 0
        while True:
            #perform action based on this policy
            a = self.predict(s, w_in, w_out, h_weights)

            if self.config.mode == 'discrete':
                a = np.argmax(a)
            s, reward, done, _ = env.step(a)

            '''when the score is summed through entire game'''
            total_reward += reward
            '''when only final score is taken'''
            #total_reward = reward

            if done:
                return total_reward
