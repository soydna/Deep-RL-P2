Both Actor and Critic uses a three layer fully connected networks with hidden layers have 200 and 150 nodes.
Leaky Relu Activation to avoid dead relu problem with normal Relu
Batch normalization is used, to the L2 Weigth decay is set to zero (https://arxiv.org/abs/1706.05350)
As suggested by Udacity gradient clipping is used for the critic.
Also the network is updated every 20 time steps, but for 10 iterations. This allows for more experience 
gathering between each learning updates which helps stablize training.
