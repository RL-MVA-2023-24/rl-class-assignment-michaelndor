from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector.async_vector_env import AsyncVectorEnv
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
# from src.env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import joblib   # Use this import instead if you have a newer version of scikit-learn

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class ProjectAgent:
    def __init__(self):
        self.Qfunction = None
        self.n_actions = 4
        state_dim = 6
        action_dim = 4
        learning_rate=1e-3
        self.q_network = QNetwork(state_dim, action_dim, hidden_size = 256)
        self.target_network = QNetwork(state_dim, action_dim, hidden_size = 256)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.action_dim = action_dim

    def act(self, observation, use_random=False):
        epsilon = 0.0
        if epsilon < np.random.random():
            network = self.model
            net_pop = self.model_pop
            device = "cuda" if next(network.parameters()).is_cuda else "cpu"
            with torch.no_grad():
                Q = network(torch.Tensor(observation).unsqueeze(0).to(device))*0.32 + net_pop(torch.Tensor(observation).unsqueeze(0).to(device))
                return torch.argmax(Q).item()
        else :
            return np.random.randint(4)
        
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
        print("Saved Q-network model.")

    def load(self):
        self.model = QNetwork(6, 4, 256)
        checkpoint = torch.load("src/agent_state0.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.load_state_dict(checkpoint) 

        self.model_pop = QNetwork(6, 4, 256)
        checkpoint_pop = torch.load("src/agent_state1.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_pop.load_state_dict(checkpoint_pop) 
        pass

    def update_network(self, experiences, gamma=0.99):
        states, actions, rewards, next_states, dones = experiences
        actions = torch.LongTensor(actions).view(-1, 1)
        state_action_values = self.q_network(torch.FloatTensor(states)).gather(1, actions)
        next_state_values = self.target_network(torch.FloatTensor(next_states)).detach().max(1)[0]
        expected_state_action_values = (next_state_values.unsqueeze(1) * gamma * (1 - torch.FloatTensor(dones))) + torch.FloatTensor(rewards)
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def create_env_fn(self, mode, env_class, max_episode_steps=200):
        def _init():
            env = TimeLimit(env_class(domain_randomization=True), max_episode_steps=max_episode_steps)
            env.reset(mode=mode)
            return env
        return _init

    def collect_samples(self, horizon, n_envs, n_reset, replay_buffer, use_Q=False):
        env_fns = [self.create_env_fn('unhealthy', env_class=HIVPatient) for _ in range(n_envs)]
        vector_env = AsyncVectorEnv(env_fns)
        
        for _ in tqdm(range(n_reset // n_envs)):
            states = vector_env.reset()[0]  # Reset returns an array for all environments
            for _ in range(horizon):
                actions = np.zeros(n_envs, dtype=np.int32)
                for env_idx in range(n_envs):
                    if not use_Q:
                        actions[env_idx] = vector_env.single_action_space.sample()
                    else:
                        state = states[env_idx]
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Assuming act requires tensor
                        action = self.act(state_tensor).item()  # Assuming act returns a tensor
                        actions[env_idx] = action
                
                next_states, rewards, dones, _, _ = vector_env.step(actions)
                # Store experiences in the replay buffer
                for env_idx, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
                    replay_buffer.append((state, action, reward, next_state, done))
                    if not dones[env_idx]:
                        states[env_idx] = next_state

    def train(self, env, horizon=200, n_reset=32, nb_episodes=1000, n_envs=8):
        replay_buffer = deque(maxlen=10000)
        update_every = 5
        batch_size = 64
        t_step = 0
        save_interval = 10

        for episode in range(nb_episodes):
            if episode == 0:
                self.collect_samples(horizon, n_envs, n_reset, replay_buffer, use_Q=False)
            else:
                self.collect_samples(horizon, n_envs, n_reset, replay_buffer, use_Q=True)

            if len(replay_buffer) > batch_size and episode % update_every == 0:
                experiences = random.sample(replay_buffer, batch_size)
                experiences = map(np.array, zip(*experiences))
                print(experiences)
                self.update_network(experiences)

            # Periodically update the target network
            if episode % (nb_episodes // 10) == 0:
                self.update_target_network()

            print(f"Episode {episode + 1} completed.")

            if episode % save_interval == 0 or episode == nb_episodes - 1:
                save_path = f"model_checkpoint.pth"
                self.save(save_path)
                
                env=TimeLimit(HIVPatient(), max_episode_steps=200)
                rewards = []
                obs, info = env.reset()
                done = False
                truncated = False
                episode_reward = 0
                while not done and not truncated:
                    action = self.act(obs)
                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    rewards.append(reward * 1e-6 * 200)

                plt.plot(rewards)
                plt.show()

        print(f"Episode {episode + 1} completed.")

# env = TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200)
# agent = ProjectAgent()
# agent.train(env)
# agent.save("agent_state0.pk")  

# env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
# agent = ProjectAgent()
# agent.train(env)
# agent.save("agent_state1.pk")  
