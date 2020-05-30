import gym

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make('LunarLander-v2')

# Instantiate the agent
model = A2C('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
# Save the agent
model.save("a2c_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = A2C.load("a2c_lunar")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()