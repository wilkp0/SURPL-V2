import gym 
import custom_env
from SBEV import SmartBuildingEnv, ChargingStationEnv
import os, sys, glob, shutil, re, random, time
from datetime import datetime
from stable_baselines3 import PPO, DDPG 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import tensorflow as tf
from tensorboard import default
from tensorboard import program
import numpy as np

start = time.time()
#-----------------------------------------------------
now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
sourceDir = "/Users/jordan/ThesisMARL/SURPL-V2/JJ_cust/"
resultsDir = sourceDir + "results/"
tbLogs = resultsDir + "tb/"
tfSBLogs = tbLogs + "SBlogs"
tfEVLogs = tbLogs + "EVlogs"
pltDir = sourceDir + "plt_figs_test"
pltDirArchive = sourceDir + "plt_figs_archive"
fileOut = open(resultsDir + "PPO.txt", "w+")

#if os.path.exists(tbLogs):
    # DIR NOT EMPTY
    #if os.listdir(tbLogs):
        #print("Results directory already exists! Cleaning... ")
        # YOU MUST COPY THE DIRECTORY MANUALLY OR ELSE YOU NEED THREADING
        #shutil.rmtree(tbLogs)
        #time.sleep(2)
        #os.mkdir(tbLogs)
        #time.sleep(1)
        #pass    

# env =  gym.make("SmartBuilding_single-v0")
#env1 =  make_vec_env("SmartBuilding_single-v0", n_envs=1)
#env2 =  make_vec_env("ChargingStation_single-v0", n_envs=1)

# print(env.observation_space)

batch_size = 10

print("Loading agents... ")

# SBE = SmartBuildingEnv()
# CSE = ChargingStationEnv()

env1 = SmartBuildingEnv()
env2 = ChargingStationEnv()

print("Loading models...")
loaded_model_SBE = PPO.load("results/PPO_SB")  
loaded_model_CSE = PPO.load("results/PPO_EV")

print("Aggregating models...")
models = [loaded_model_SBE, loaded_model_CSE]
envs = [env1, env2]

print("Iterating...")
#---------

modelNames = ["SB", "EV"]

for model, env, nm in zip(models, envs, modelNames):
    #model.learn(total_timesteps=ep_per_batch, reset_num_timesteps=False, tb_log_name=tfSBLogs, ) if nm == "SB" else model.learn(total_timesteps=ep_per_batch, reset_num_timesteps=False, tb_log_name=tfEVLogs)
    #time.sleep(7)
    #model.save("results/PPO_" + nm)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}", file=fileOut)
    

    obs = env.reset() 
    for i in range(batch_size):
        action, _states = model.predict(obs, deterministic=True)
        
  
        obs, reward, done, info = env.step(action)

        env.render() 

        if done:
            obs = env.reset()

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