import gymnasium as gym
from math import pi

env = gym.make('CartPole-v1', render_mode="human")
env.reset()

myAction = 0

while True:
    env.render()
    # action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(myAction)

    cartPosition = observation[0]
    cartVelocity = observation[1]
    poleAngle = observation[2]
    poleAngleVelocity = observation[3]
    print(poleAngle)

    if poleAngle < 0:
        myAction = 0
    else:
        myAction = 1

    if terminated:
        # env.reset()
        break
    if truncated:
        break
env.close()
