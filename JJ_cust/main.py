import gym 
import custom_env
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
sourceDir = "/Users/jordan/ThesisMARL/SURPL-V2/JJ_cust"
fileOut = open(sourceDir + "/results/PPO.txt", "w+")


env =  gym.make("SmartBuilding_single-v0")

max_ep = 10000

for episodeCount in range(max_ep):
    stepCount = 0 
    epReward = 0
    done = False
    state = env.reset()
    
    while not done: 
        observation, reward, done, _ = env.step(env.action_space.sample())
        # nextState, reward, done, _ = env.step()
        env.render()
        stepCount += 1
        epReward +=  reward 
        state  = observation
        
    print("Episode: {}, Step Count: {}, Episode Reward: {}".format(episodeCount, stepCount, epReward))
    print()
    print("Episode: {}, Step Count: {}, Episode Reward: {}".format(episodeCount, stepCount, epReward), file=fileOut)
    print(file=fileOut)

env.close()