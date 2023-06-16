from tqdm import tqdm
import os, shutil
import glob, imageio
import cv2
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from tetris_env_3 import TetrisEnv

NUM_ENV = 100
MODEL_PATH = 'env3_model/best_model.zip'

vec_env = make_vec_env(TetrisEnv, n_envs=NUM_ENV)
obs = vec_env.reset()
model = A2C.load(MODEL_PATH, vec_env, custom_objects = {"observation_space": vec_env.observation_space, "action_space": vec_env.action_space})

replay_folder = './replay'
if os.path.exists(replay_folder):
    shutil.rmtree(replay_folder)

test_steps = 5000
n_env = obs.shape[0]
ep_id = np.zeros(n_env, dtype=int)
cum_reward = np.zeros(n_env)
ep_steps = np.zeros(n_env, dtype=int)
max_reward = 0
max_game_id = 0
max_ep_id = 0
max_rm_lines = 0
max_lifetime = 0

for step in tqdm(range(test_steps)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
        
    for eID in range(n_env):
        cum_reward[eID] += reward[eID]
        folder = f'{replay_folder}/{str(eID).zfill(4)}/{str(ep_id[eID]).zfill(4)}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        fname = f'{folder}/{str(ep_steps[eID]).zfill(4)}.png'
        cv2.imwrite(fname, np.transpose(obs[eID], (1, 2, 0)))
        ep_steps[eID] += 1
        
        if done[eID]:
            if cum_reward[eID] > max_reward:
                max_reward = cum_reward[eID]
                max_game_id = eID
                max_ep_id = ep_id[eID]
                max_rm_lines = info[eID]['removed_lines']
                max_lifetime = info[eID]['life_time']
                
            ep_id[eID] += 1
            cum_reward[eID] = 0
            ep_steps[eID] = 0

best_replay_path = f'{replay_folder}/{str(int(max_game_id)).zfill(4)}/{str(int(max_ep_id)).zfill(4)}'
print(f'After playing 30 envs each for {test_steps} steps:')
print(f'Max reward: {max_reward}, Best video: {best_replay_path}')
print(f'Removed lines: {max_rm_lines}, lifetime: {max_lifetime}')

filenames = glob.glob(f'{best_replay_path}/*.png')
images = []
for filename in sorted(filenames):
    images.append(imageio.v2.imread(filename))
imageio.mimsave('replay.gif', images, loop=0)

vec_env.close()

if os.path.exists(replay_folder):
    shutil.rmtree(replay_folder)