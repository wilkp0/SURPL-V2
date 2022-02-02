import gym 
import custom_env
from SBEV import SmartBuildingEnv
# ChargingStationEnv
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
# sourceDir = "/Users/jordan/ThesisMARL/SURPL-V2/JJ_cust"
# fileOut = open(sourceDir + "/results/PPO.txt", "w+")

# env =  gym.make("SmartBuilding_single-v0")
env =  make_vec_env("SmartBuilding_single-v0", n_envs=1)

# print(env.observation_space)

batch_size = 3
ep_per_batch = 5

print("Loading agents... ")

SBE = SmartBuildingEnv()
# CSE = ChargingStationEnv()

print("Loading models...")
model_SBE = PPO('MlpPolicy', env, verbose = 1)
# model_CSE = PPO('MlpPolicy', CSE, verbose = 1)

print("Aggregating models...")
# models = [model_SBE, model_CSE]

print("Iterating...")
# for i in range(batch_size):
#     for model in models:
#         action, _states = model.predict(obs, deterministic=True)
#         model.learn(total_timesteps=ep_per_batch)
#         obs = env.reset()

# for model in models:
#     model.learn(total_timesteps=ep_per_batch)
#     obs = env.reset()
#     for i in range(batch_size):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         if done: 
#             obs = env.reset()

model_SBE.learn(total_timesteps=ep_per_batch)
obs = env.reset()
for i in range(batch_size):
    action, _states = model_SBE.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
        


env.close()