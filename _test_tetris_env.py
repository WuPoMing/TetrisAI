import numpy as np
import cv2
import matplotlib.pyplot as plt
from tetris_env_4 import TetrisEnv

env = TetrisEnv()
obs, info = env.reset()

action_list = []
for _ in range(1000):
    action = env.action_space.sample()
    # action = 4
    action_list.append(int(action))
    obs, reward, terminated, truncated, info = env.step(action)
    env.render(pause=True)

    if terminated:
        break

print(action_list)
env.close()
plt.show()