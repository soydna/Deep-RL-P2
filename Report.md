## Architecture
Both Actor and Critic uses a fully connected networks with two hidden layers have 200 and 150 nodes.  
Leaky Relu Activation to avoid dead relu problem with normal Relu.  
Batch normalization is used, to the L2 Weigth decay is set to zero (https://arxiv.org/abs/1706.05350).  
As suggested by Udacity gradient clipping is used for the critic.  
Also the network is updated every 20 time steps, but for 10 iterations. This allows for more experience gathering between each learning updates which helps stablize training.

## Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size   
BATCH_SIZE = 64         # minibatch size   
GAMMA = 0.99            # discount factor  
TAU = 1e-3              # for soft update of target parameters   
LR_ACTOR = 1e-4         # learning rate of the actor   
LR_CRITIC = 3e-4        # learning rate of the critic  
WEIGHT_DECAY = 0.0      # L2 weight decay  


## Episodes to reach moving avreage (100) of target of 30.0
350


## Future Work
* Try the second version with 20 concurent agents and see how that trains and compare the number of episodes required
* Try other RL aproaches as A2C and D4PG
