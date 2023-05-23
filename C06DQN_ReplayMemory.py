import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import random
from collections import deque


# 定义神经网络模型
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DQN算法类
class DQNAgent:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, lr=0.001, memory_capacity=10000, batch_size=64):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size

        self.model = DQN(input_size, output_size)
        self.target_model = DQN(input_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=self.memory_capacity)

    def select_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.tensor([[random.randint(0, self.output_size - 1)]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.cat(state_batch)
        action_batch = torch.cat(action_batch)
        reward_batch = torch.cat(reward_batch)
        next_state_batch = torch.cat(next_state_batch)
        done_batch = torch.cat(done_batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.sample_batch()

        q_values = self.model(state_batch).gather(1, action_batch)
        next_q_values = self.target_model(next_state_batch).max(1)[0].detach()

        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# 创建CartPole环境
env = gym.make('CartPole-v1', render_mode=None)
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# 创建DQN代理
agent = DQNAgent(input_size, output_size)

# 训练DQN代理
episodes = 300
for episode in range(episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action.item())
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        agent.store_transition(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward.item()

    # 更新目标网络
    if episode % 10 == 0:
        agent.update_target_model()

    # 衰减探索率
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# 测试训练好的模型
env.close()
env = gym.make('CartPole-v1', render_mode="human")

while True:
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action.item())
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)

        state = next_state
        total_reward += reward.item()

    print(f"Test Total Reward: {total_reward}")
