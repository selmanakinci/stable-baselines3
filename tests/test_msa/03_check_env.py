# check environment
import gym
import gym_ransim
#from ransim_env import RanSimEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

#env = RanSimEnv()
env = gym.make('ransim-v0')
#env = gym.make('CartPole-v1')

# Define and Train the agent
model = A2C('MlpPolicy', env).learn(total_timesteps=1000)

# It will check your custom environment and output additional warnings if needed
check_env(env)