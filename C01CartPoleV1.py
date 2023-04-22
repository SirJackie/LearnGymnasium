import gym

env = gym.make('CartPole-v1', render_mode="human")
env.reset()

while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, done, info = env.step(action)
    if terminated:
        env.reset()
    if done:
        break
env.close()
