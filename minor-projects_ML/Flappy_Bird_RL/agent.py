import flappy_bird_gymnasium
import gymnasium as gym
from DQN import dqn
import torch
import torch.nn as nn
from experience_replay import ReplayMemory
import itertools

# detect gpu and set gpu 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


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
        
        while not terminated:
            action = env.action_space. sample()

            
            next_state, reward, terminated, _, _ = env.step(action)

            if is_training:
                memory.append((state, action, next_state, reward, terminated))

            # Checking if the payer is still alive
            if terminated:
                break

    #env.close() - manually stop

    