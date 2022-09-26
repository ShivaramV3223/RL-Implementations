import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from VPGpolicy import *

class VPGAgent:
    def __init__(self,agent_params, env):
        self.gamma = agent_params['Gamma']
        self.lr = agent_params['LR']
        self.episodes = agent_params['num_episodes']
        self.policy_params = agent_params['policy_params']
        self.value_params = agent_params['value_params']
        self.max_timesteps = agent_params['num_timesteps']
        self.env = env
        self.rewards = []
        self.observations = []
        self.actions = []
        self.loss = []
        self.lossfn = nn.MSELoss()
        self.policy = Vanilla_Policy_Gradient(self.policy_params)
        self.valuefn = Value_function(self.value_params)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr =self.lr)
        self.optim_value = torch.optim.Adam(self.valuefn.parameters(), lr = self.lr)
############################## Calculating discounted cumulative sum#######################################################################
    def discounted_cum_sum(self, rewards):
        rew = np.array(rewards)
        dicounted_sum = []
        rew_mat = np.full(len(rewards), rew)
        po = np.power(np.full(len(rewards), self.gamma), np.arange(len(rewards)))
        gamma_mat = np.triu(np.full(len(rew), po))/po[:None]
        discounted_sum = np.sum(np.multiply(rew_mat, gamma_mat), 1)
        return discounted_sum
############################### Function to calculate the reward return################################################################
    def reward_return(self, epi = -1):
        returns = 0
        if epi == -1:
            returns = []
            for episodes in range(len(self.rewards)):
                pow_gamma = np.power(np.full(len(self.rewards[episodes]), self.gamma), np.arange(len(self.rewards[episodes])))
                ret = np.sum(pow_gamma*np.array(self.rewards[episodes]))
                returns.append(ret)
        else:
            pow_gamma = np.power(np.full(len(self.rewards[epi]), self.gamma), np.arange(len(self.rewards[epi])))
            returns = np.sum(pow_gamma*np.array(self.rewards[epi]))
        return returns
    
################################# Training the Value Function for every Episode with the obtained rewards####################################
    def value_func_train(self, rewards, observations):
        for timestep in range(len(rewards) - 1):
            value_esti = self.valuefn(torch.from_numpy(observations[timestep]).to(torch.float32))
            target = rewards[timestep]+ self.gamma * self.valuefn(torch.from_numpy(observations[timestep + 1]).to(torch.float32))
            loss = self.lossfn(value_esti, target)
            self.optim_value.zero_grad()
            loss.backward()
            self.optim_value.step()
###################################### Update for the policy network###########################################################################
    def backward(self, log_probs, rewards, observations):
        Q = self.discounted_cum_sum(self.rewards[-1])
        Q = torch.from_numpy(Q).to(torch.float32)
        #log_probs = torch.cat(log_probs).reshape(len(self.rewards[-1]), -1)
        log_probs = torch.tensor(log_probs)
        B = []
        for timestep in range(len(observations)):
            B.append(self.valuefn(torch.from_numpy(observations[timestep]).to(torch.float32)))
        B = torch.tensor(B)
        A = Q - B[:-1]
        loss = -(log_probs*A)
        loss.requires_grad_(True)
        loss = loss.mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.loss.append(loss.detach().numpy())                                          
##################################################################### Traing the policy network########################################################
    def train(self, new = True):
        if not new:
            self.policy.load_state_dict(torch.load('model_weights.pth'))
            
        for episodes in range(self.episodes):
            #print(episodes)
            observation = self.env.reset()
            observations = [observation]
            log_probs = []
            actions = []
            rewards = []
            for time_step in range(self.max_timesteps):
                observation = torch.from_numpy(observation).to(torch.float32)
                distribution = self.policy(observation)
                action = distribution.sample()
                actions.append(action)
                observation, reward, done, info = self.env.step(action)
                observations.append(observation)
                log_probs.append(distribution.log_prob(action))
                rewards.append(reward)
                # if done:
                #     break
            self.rewards.append(rewards)
            #self.observations.append(observations)
           # self.actions.append(actions)
            self.value_func_train(rewards, observations)
            self.backward(log_probs, rewards, observations)
            # print("--------------------------------")
            # print(f"Episode: {episodes}")
            # print("--------------------------------")
            # print(f"loss : {self.loss[-1]}")
            # print(f"Return : {self.reward_return(episodes)}")
        torch.save(self.policy.state_dict(), 'model_weights.pth')
        return (self.observations, self.actions, self.rewards, self.loss)
################################################################### To evaluate the trained policy ########################################################
    def eval(self, episodes):
        self.policy.load_state_dict(torch.load('model_weights.pth'))
        self.policy.eval()
        rewards = []
        observation = self.env.reset()
        with torch.no_grad():
            for episode in range(episodes):
                self.env.render()
                observation = torch.from_numpy(observation).to(torch.float32)
                distribution = self.policy(observation)
                action = distribution.sample()
                observation, reward, done, info = self.env.step(action)
                rewards.append(reward)
        return rewards
########################################################### Plotting the episodes vs Loss and Return graphs ##############################################
    def plot(self, type='raw'):
        episodes = range(len(self.loss))
        if type == 'raw':
            returns = self.reward_return()
        elif type == 'rung_avg':
            returns = self.reward_return()
            sum_ = 0
            returns = [(sum_+returns[i])/(i+1)  for i in range(len(returns))]
        #print('returns', returns)
        #print('loss',self.loss)
        figure, axis = plt.subplots(2,1)
        axis[0].plot(episodes, returns)
        axis[0].set_title("Episodes vs Returns")
        axis[0].set(xlabel='Episodes', ylabel='Returns')

        axis[1].plot(episodes, self.loss)
        axis[1].set_title("Episodes vs Loss", y=-0.001)
        axis[1].set(xlabel='Episodes', ylabel='Log score')

        plt.show()
