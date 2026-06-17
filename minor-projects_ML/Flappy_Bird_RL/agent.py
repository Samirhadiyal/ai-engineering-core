import flappy_bird_gymnasium
import gymnasium as gym
from DQN import dqn
import torch
import torch.nn as nn
import torch.optim as optim
from experience_replay import ReplayMemory
import itertools
import yaml

# detect gpu and set gpu 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set
        
        with open("parameters.yaml", 'r') as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]
            
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        
        
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        
        
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
            
    def run(self, is_training=True, render = False):
    
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0] # input dimensions
        num_actions = env.action_space.n # output  dimensions
        
        policy_dqn = DQN(num_states, num_actions).to(device)
        
        
        if is_training:
            memory = ReplayMemory(10000)
        
        for episode in itertools.count():
            state, _ = env.reset()
            episode_rewards = 0
            terminated = False
            
            while not terminated:
                action = env.action_space. sample()

                
                next_state, reward, terminated, _, _ = env.step(action)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))

                state = new_state
                episode_rewards += rewards
            
            print(f"episode={episode+1}, with total reward={episode_rewards}")
        #env.close() - manually stop
    
    