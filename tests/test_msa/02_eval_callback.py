import os
import gym
import gym_ransim
import matplotlib.pyplot as plt

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CustomRansimCallback
#from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# environments setup
#env = gym.make('Pendulum-v0')
#env = Monitor(env, log_dir)
#env_string = 'LunarLander-v2'
env_string = 'ransim-v0'
env = make_vec_env(env_string, env_kwargs={"t_final": 200},  n_envs=2,  monitor_dir=log_dir)  # Parallel environments



eval_env = gym.make(env_string, t_final=105)

# Use deterministic actions for evaluation
'''eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=200,
                             deterministic=True, render=False)'''

ransim_callback = CustomRansimCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=2*1e2,
                             deterministic=True, render=False,
                             plot_results=True)

model = A2C('MlpPolicy', env, verbose=1)
timesteps = 4*2*1e2  # k*n_envs*T_Final
model.learn(total_timesteps=int(timesteps), callback=ransim_callback)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "A2C ran-sim")
plt.savefig(log_dir + 'A2C_ran-sim_rewards_plot.png', format="png")
plt.show()

msa = 1
