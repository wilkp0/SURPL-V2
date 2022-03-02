import gym 
import custom_env
from SBEV import SmartBuildingEnv, ChargingStationEnv
import os, sys, glob, shutil, re, random, time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import tensorflow as tf
from tensorboard import default
from tensorboard import program
import numpy as np

start = time.time()
#------------------------------------------------------------
now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
sourceDir = "/Users/jordan/ThesisMARL/SURPL-V2/JJ_cust"
resultsDir = sourceDir + "/results/"
tbLogs = resultsDir + "/tb/"
tfSBLogs = tbLogs + "SBlogs"
tfEVLogs = tbLogs + "EVlogs"
fileOut = open(sourceDir + "/results/PPO.txt", "w+")

if os.path.exists(tbLogs):
    # DIR NOT EMPTY
    if os.listdir(tbLogs):
        print("Results directory already exists! Cleaning... ")
        # YOU MUST COPY THE DIRECTORY MANUALLY OR ELSE YOU NEED THREADING
        shutil.rmtree(tbLogs)
        time.sleep(2)
        os.mkdir(tbLogs)
        time.sleep(1)
        pass    
    

# print(tfSBLogs)
# print(tfEVLogs)

# env1 =  make_vec_env("SmartBuilding_single-v0", n_envs=1)
# env2 =  make_vec_env("ChargingStation_single-v0", n_envs=1)

# batch_size = 1000
# ep_per_batch = 50 * batch_size
batch_size = 2000
ep_per_batch = 50 * batch_size

print("Loading agents... ")

# SBE = SmartBuildingEnv()
# CSE = ChargingStationEnv()
env1 = SmartBuildingEnv()
env2 = ChargingStationEnv()

print("Loading models...")
model_SBE = PPO('MlpPolicy', env1, verbose = 1, tensorboard_log=tfSBLogs)
model_CSE = PPO('MlpPolicy', env2, verbose = 1, tensorboard_log=tfEVLogs)

print("Aggregating models...")
models = [model_SBE, model_CSE]
envs = [env1, env2]

# print("Checking envs...")
# for e in envs:
#     check_env(e)

print("Iterating...")
#---------

modelNames = ["SB", "EV"]

for model, env, nm in zip(models, envs, modelNames):
    model.learn(total_timesteps=ep_per_batch, reset_num_timesteps=False, tb_log_name=tfSBLogs) if nm == "SB" else model.learn(total_timesteps=ep_per_batch,  reset_num_timesteps=False, tb_log_name=tfEVLogs)
    # model.learn(total_timesteps=ep_per_batch, eval_log_path=sourceDir + "/results/",  reset_num_timesteps=False, tb_log_name=tfSBLogs) if nm == "SB" else model.learn(total_timesteps=ep_per_batch, eval_log_path=sourceDir + "/results/",  reset_num_timesteps=False, tb_log_name=tfEVLogs)
    time.sleep(7)
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}", file=fileOut)
    
    time.sleep(7)
    model.save("results/PPO_" + nm)
    
    obs = env.reset() 
    for i in range(batch_size):
        action, _states = model.predict(obs, deterministic=True)
        
        # action = np.clip(action, env.action_space.low, env.action_space.high)
        
        obs, reward, done, info = env.step(action)
        # if ep_per_batch % 2000 == 0:
        # env.render()
        if done:
            obs = env.reset()
# for model, env in zip(models, envs):
#     mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
#     print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
#     print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

#wait for 58 minutes
# time.sleep(3500)


#------------------------------------------------------------
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("-"*25)
print("-"*25)
print("-"*25, file=fileOut)
print("-"*25, file=fileOut)
print("Progran runtime: \n {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes), seconds))
print("Progran runtime: \n {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes), seconds))

fileOut.close()
# env.close()