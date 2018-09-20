import numpy as np

from GA_Config import Config
from GA_Network import Network

config = Config()

#copies all the weights to a another network object
def copy_net(old_net):
    new_net = Network(config)
    new_net.w_in = np.copy(old_net.w_in)
    for i in range(config.num_layers):
        new_net.weights[i] = np.copy(old_net.weights[i])
    new_net.w_out = np.copy(old_net.w_out)
    return new_net

#weights are mutated by adding random noise
def mutation(policy):
    noise = np.random.randn(policy.total_num)
    noise *= config.sigma

    policy.w_in += noise[:policy.num_in].reshape(policy.w_in.shape)
    curr = policy.num_in
    for i in range(config.num_layers):
        policy.weights[i] += noise[curr:curr+policy.num_weights[i]].reshape(policy.weights[i].shape)
        curr += policy.num_weights[i]
    policy.w_out += noise[policy.total_num - policy.num_out:].reshape(policy.w_out.shape)
