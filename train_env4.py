from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import A2C
from tetris_env_4 import TetrisEnv

EXP_NAME = 'env4'
NUM_ENV = 32
TOTAL_TIMESTEPS = int(100000000)
SAVE_DIR = f'{EXP_NAME}_model'

env = make_vec_env(TetrisEnv, n_envs=NUM_ENV)
callback = EvalCallback(env, best_model_save_path=SAVE_DIR, log_path=SAVE_DIR, 
                        eval_freq=500, 
                        deterministic=True, 
                        render=False
                        )
model = A2C('CnnPolicy', env, tensorboard_log='logs', verbose=1)
model.learn(total_timesteps=TOTAL_TIMESTEPS, 
            callback=callback, 
            tb_log_name='A2C4', 
            reset_num_timesteps=True
            )
env.close()