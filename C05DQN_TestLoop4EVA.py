import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym


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
    def __init__(self, input_size, output_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, lr=0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr

        self.model = DQN(input_size, output_size)
        self.target_model = DQN(input_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.tensor([[torch.randint(self.output_size, ())]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1)

    def train(self, state, action, reward, next_state, done):
        q_value = self.model(state).gather(1, action)
        next_q_value = self.target_model(next_state).max(1)[0].detach()

        expected_q_value = reward + (1 - done) * self.gamma * next_q_value

        loss = self.loss_fn(q_value, expected_q_value.unsqueeze(1))

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

        agent.train(state, action, reward, next_state, done)

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
