import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
import random
from tqdm import tqdm
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_height = 125
img_width = 400
img_transforms = [
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_transforms)

class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

        self.waypoint = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

        del self.waypoint[:]

class ActorCritic(nn.Module):
    def __init__(self, critic_state_dim, actor_input_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor_conv =  nn.Sequential(
                nn.Conv2d(actor_input_dim, 64,  5, stride=3, padding=2), nn.LeakyReLU(),nn.MaxPool2d(2, 2),
                nn.Conv2d(64,               128, 5, stride=4, padding=2), nn.LeakyReLU(),nn.MaxPool2d(2, 2),
                nn.Conv2d(128,              64,  3, stride=2, padding=1), nn.MaxPool2d(2, 2)
                )
        self.actor_mlp = nn.Sequential(
                nn.Linear(128, 64), nn.LeakyReLU(),
                nn.Linear(64, 32), nn.LeakyReLU(),
                nn.Linear(32, 1), nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(critic_state_dim, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 32), nn.ReLU(),
                nn.Linear(32, 1) 
        )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory, waypoint, is_test):
        state_cuda    = state.to(device)
        action_middle = self.actor_conv(state_cuda)
        action_middle = action_middle.view(-1, 128)
        action_mean   = self.actor_mlp(action_middle)
        # print(action_mean)
        # action_mean = self.actor(state)
        if is_test == False:
            cov_mat = torch.diag(self.action_var).to(device)
            
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            # print(action_logprob)

            memory.states.append(state.detach().cpu())
            memory.actions.append(action.detach().cpu())
            memory.logprobs.append(action_logprob.detach().cpu())
            memory.waypoint.append(waypoint.detach().cpu())
        
        return action.detach() if is_test == False else action_mean.detach()
    
    def evaluate(self, state, action, waypoint):   
        # action_mean = self.actor(state)
        action_middle = self.actor_conv(state)
        action_middle = action_middle.view(-1, 128)
        action_mean   = self.actor_mlp(action_middle)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(waypoint)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO(object):
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy = ActorCritic(critic_state_dim=state_dim, action_std=action_std,actor_input_dim=3, action_dim=1).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        # self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old = ActorCritic(critic_state_dim=state_dim, action_std=action_std,actor_input_dim=3, action_dim=1).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory, waypoint, is_test=False):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        img = Image.fromarray(cv2.cvtColor(state,cv2.COLOR_BGR2RGB))
        state = img_trans(img).unsqueeze(0)
        waypoint = torch.FloatTensor(waypoint.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory, waypoint, is_test).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards_np = np.array(rewards)
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor


        # old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        # old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()

        # old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # old_waypoint = torch.squeeze(torch.stack(memory.waypoint), 1).to(device).detach()

        batch_size = 256
        # Optimize policy for K epochs:
        # for _ in range(self.K_epochs):
        for i in tqdm(range(self.K_epochs), desc='Update policy'):

            index = np.random.choice(len(memory.actions), size=batch_size)

            states = np.array([t.squeeze(0).numpy() for t in memory.states])
            actions = np.array(memory.actions)
            logprobs = np.array(memory.logprobs)
            waypoints = np.array([w.squeeze(0).numpy() for w in memory.waypoint])

            batch_states = states[index]
            batch_actions = actions[index]
            batch_logprobs = logprobs[index]
            batch_waypoints = waypoints[index]
            batch_rewards = rewards_np[index]

            old_states = torch.from_numpy(batch_states).to(device)
            old_actions = torch.from_numpy(batch_actions).unsqueeze(1).to(device)
            old_logprobs = torch.from_numpy(batch_logprobs).to(device)
            old_waypoint = torch.from_numpy(batch_waypoints).to(device)
            rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_waypoint)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v3"
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 2           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 2000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break
        
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
                
            # index = np.random.choice(len(memory.actions), size=batch_size)

            # states = np.array(memory.states)
            # actions = np.array(memory.actions)
            # logprobs = np.array(memory.logprobs)
            # waypoints = np.array(memory.waypoint)

            

            # states_tensor = torch.FloatTensor(states[index])
            # actions_tensor = torch.FloatTensor(actions[index])
            # logprobs_tensor = torch.FloatTensor(logprobs[index])
            # waypoints_tensor = torch.FloatTensor(waypoints[index])

            # print(actions_tensor)
            
            # old_states = torch.squeeze(torch.stack(states_tensor).to(device), 1).detach()
            # old_actions = torch.squeeze(torch.stack(actions_tensor).to(device), 1).detach()
            # old_logprobs = torch.squeeze(torch.stack(logprobs_tensor), 1).to(device).detach()

            # old_waypoint = torch.squeeze(torch.stack(waypoints_tensor), 1).to(device).detach()
