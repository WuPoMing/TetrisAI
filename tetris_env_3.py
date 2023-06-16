import socket
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C
import imageio

class TetrisEnv(gym.Env):
	metadata = {'render_modes': ['human'], 'render_fps': 30}
	action_space = spaces.Discrete(5)
	observation_space = spaces.Box(low=0, high=255, shape=(3, 200, 100), dtype=np.uint8)
	def __init__(self):
		super().__init__()
		self._init()
		self.adress = ('127.0.0.1', 10612)
		self.game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.game.connect(self.adress)
		
	def _init(self):
		self.move_left_counter = 0
		self.move_right_counter = 0
		self.rotate_counter = 0
		self.block_height = 17

		self.last_holes = 0
		self.last_bumps = 0
		self.last_height = 0
		self.last_removed_lines = 0

		self.life_time = 0
		self.total_reward = 0

	def step(self, action):
		if action == 0:
			self.game.sendall(b'move -1\n')
			self.move_left_counter += 1
		elif action == 1:
			self.game.sendall(b'move 1\n')
			self.move_right_counter += 1
		elif action == 2:
			self.game.sendall(b'rotate 0\n')
			self.rotate_counter += 1
		elif action == 3:
			self.game.sendall(b'rotate 1\n')
			self.rotate_counter += 1
		elif action == 4:
			self.game.sendall(b'drop\n')
		#########################################################################################
		terminated, self.removed_lines, height, holes, self.observation, bump_count, new_block = self._get_info()
		# update values
		if action != 0 or new_block:
			self.move_left_counter = 0
		if action != 1 or new_block:
			self.move_right_counter = 0
		if (action != 2 and action != 3) or new_block:
			self.rotate_counter = 0
		if new_block:
			self.block_height = 17
		else:
			self.block_height -= 1
		#########################################################################################
		if self.move_left_counter > 7 or self.move_right_counter > 4 or self.rotate_counter > 3:
			terminated = True
		# bonus reward
		if self.block_height > 10:
			reward = (0.2 * (self.block_height - 10) ** 2) * 0.1
		else:
			reward = 0.0

		if new_block:
			# 1
			hole_delta = holes - self.last_holes
			self.last_holes = holes
			reward += (-10 * hole_delta) * 0.1
			# 2
			height_delta = height - self.last_height
			self.last_height = height
			reward += (-5 * height_delta) * 0.1
			# 3
			bump_delta = bump_count - self.last_bumps
			self.last_bumps = bump_count
			if bump_count < 6 and bump_delta > 0:
				bump_delta = 0
			reward += (-2.5 * bump_delta) * 0.1
			# 4
			score_delta = self.removed_lines - self.last_removed_lines
			self.last_removed_lines = self.removed_lines
			reward += 1000 * score_delta * 0.1

		self.life_time += 1
		self.total_reward += reward
		info = {'total_reward': self.total_reward, 'removed_lines': self.removed_lines, 'life_time': self.life_time}
		return (self.observation, reward, terminated, False, info)

	def reset(self):
		self.game.sendall(b'start\n')
		terminated, self.removed_lines, height, holes, self.observation, bump_count, new_block = self._get_info()
		self._init()
		info = {'total_reward': self.total_reward, 'removed_lines': self.removed_lines, 'life_time': self.life_time}
		return self.observation, info
	
	def render(self, pause=True):
		plt.imshow(self.observation)
		if pause == True:
			plt.pause(0.000001)

	def close(self):
		self.game.close()
	
	def _get_info(self):
		is_game_over = (self.game.recv(1) == b'\x01')
		removed_lines = int.from_bytes(self.game.recv(4), 'big')
		height = int.from_bytes(self.game.recv(4), 'big')
		holes = int.from_bytes(self.game.recv(4), 'big')
		img_size = int.from_bytes(self.game.recv(4), 'big')
		img_png = self.game.recv(img_size)

		nparr = np.frombuffer(img_png, np.uint8)
		np_image = cv2.imdecode(nparr, -1)

		gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)[::10, ::10]
		binary_image = np.where(gray_image > 0, 255, 0)
		# True/False
		new_block = binary_image[0, 6] == 255
		# bump_count
		highest_block_pos = [0] * 10
		bump_count = 0
		for x in range(10):
			col_x = binary_image[20 - height:, x]
			if np.all(col_x == 0):
				highest_block_pos[x] = 0
			else:
				highest_block_pos[x] = height - np.argmax(col_x)
		for i in range(1, 10):
			bump_count += abs(highest_block_pos[i] - highest_block_pos[i-1])
		np_image = np.transpose(np_image, (2, 0, 1))	# (3, 200, 100)

		return is_game_over, removed_lines, height, holes, np_image, bump_count, new_block

if __name__ == '__main__':

	def display(obs_list):
		for obs in obs_list:
			obs = cv2.resize(obs, (400, 800), interpolation=cv2.INTER_NEAREST)
			cv2.imshow('Tetris', obs)
			cv2.waitKey(100)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	MODEL_PATH = 'env3_model/best_model.zip'

	env = TetrisEnv()
	obs, info = env.reset()
	model = A2C.load(MODEL_PATH, env, custom_objects = {"observation_space": env.observation_space, "action_space": env.action_space})

	action_list = []
	obs_list = []
	for _ in range(1000):
		action, info = model.predict(obs, deterministic=True)
		action_list.append(int(action))
		obs, reward, done, _, info = env.step(action)
		obs_list.append(np.transpose(obs, (1, 2, 0)))
		# env.render(pause=False)

		if done:
			display(obs_list)
			imageio.mimsave('env3.gif', obs_list, loop=0)
			print(f'Total Reward: {info["total_reward"]}')
			print(f'Removed lines: {info["removed_lines"]}, Life Time: {info["life_time"]}')
			print(action_list)
			break

	plt.show()
	env.close()