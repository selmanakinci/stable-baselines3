# learn, save, load, evaluate
import gym
import gym_ransim
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
#env = gym.make('LunarLander-v2')
env = gym.make('ransim-v0')

# Instantiate the agent
model = A2C('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e3), eval_log_path= 'log_msa')
# Save the agent
model.save("a2c_ran")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = A2C.load("a2c_ran")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)
episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=True)
#print('mean_reward: %.3f  std_reward: %.3f' %(mean_reward, std_reward))
msa =1
'''# Enjoy trained agent
obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()'''