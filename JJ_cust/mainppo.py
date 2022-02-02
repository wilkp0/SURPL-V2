import gym 
import custom_env
from SBEV import SmartBuildingEnv
# ChargingStationEnv
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time

# now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
# sourceDir = "/Users/jordan/ThesisMARL/SURPL-V2/JJ_cust"
# fileOut = open(sourceDir + "/results/PPO.txt", "w+")

# env =  gym.make("SmartBuilding_single-v0")
env1 =  make_vec_env("SmartBuilding_single-v0", n_envs=1)
env2 =  make_vec_env("ChargingStation_single-v0", n_envs=1)

# print(env.observation_space)

batch_size = 3
ep_per_batch = 5

print("Loading agents... ")

# SBE = SmartBuildingEnv()
# CSE = ChargingStationEnv()

print("Loading models...")
model_SBE = PPO('MlpPolicy', env1, verbose = 1)
model_CSE = PPO('MlpPolicy', env2, verbose = 1)

print("Aggregating models...")
models = [model_SBE, model_CSE]
envs = [env1, env2]

print("Iterating...")
# for i in range(batch_size):
#     for model in models:
#         action, _states = model.predict(obs, deterministic=True)
#         model.learn(total_timesteps=ep_per_batch)
#         obs = env.reset()

# WORKS------
# model_SBE.learn(total_timesteps=ep_per_batch)
# obs = env1.reset()
# for i in range(batch_size):
#     action, _states = model_SBE.predict(obs, deterministic=True)
#     obs, reward, done, info = env1.step(action)
#     if ep_per_batch % 2000 == 0:
#         env1.render()
#     if done:
#       obs = env1.reset()
#---------

modelNames = ["SB", "EV"]

for model, env, nm in zip(models, envs, modelNames):
    model.learn(total_timesteps=ep_per_batch, eval_log_path="/Users/jordan/ThesisMARL/SURPL-V2/JJ_cust/results/")
    time.sleep(5)
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    time.sleep(10)
    model.save("results/PPO_" + nm)
    obs = env.reset() 
    for i in range(batch_size):
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        # if ep_per_batch % 2000 == 0:
        env.render()
        if done:
            obs = env.reset()
# for model, env in zip(models, envs):
#     mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
#     print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
#     print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
            
time.sleep(3500)

# model_SBE.learn(total_timesteps=ep_per_batch)
# obs = env1.reset()
# for i in range(batch_size):
#     action, _states = model_SBE.predict(obs, deterministic=True)
#     obs, reward, done, info = env1.step(action)
#     if ep_per_batch % 2000 == 0:
#         env1.render()
#     if done:
#       obs = env1.reset()



# env.close()