import os

import gym
import gym_ransim
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (BaseCallback, CustomRansimCallback, EvalCallback)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('ransim-v0')
#env = Monitor(env, log_dir)

'''
# Add some action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# Because we use parameter noise, we should use a MlpPolicy with layer normalization
model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=0)'''
# Instantiate the agent
model = A2C('MlpPolicy', env, verbose=1)

# Create the callback: check every 1000 steps
#callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
#ransim_callback = CustomRansimCallback(env, save_freq=1000, save_path=log_dir)
#eval_env = gym.make('ransim-v0')
eval_env = gym.make('LunarLanderContinuous-v2')
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=500)
'''
# run baseline algorithm
baseline_score = 0
done = False
observation = env.reset(NO_logging=0)
while not done:
    action = 'baseline'
    observation_, reward, done, info = env.step(action)
    #if done:
    #    env.plot()
    observation = observation_
    baseline_score += reward
print('baseline score: %.3f' % baseline_score)'''

# Train the agent
timesteps = 1*5000 #1e5
#model.learn(total_timesteps=int(timesteps), callback=ransim_callback)
model.learn(total_timesteps=int(timesteps))

#fig = plt.figure( )
plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "A2C ran-sim")
plt.savefig(log_dir + 'A2C_ran-sim_rewards_plot.png', format="png")
plt.show()



episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=True)
