from gym import Env
from gym.spaces import Discrete, Box
from gym.envs.classic_control import rendering
from datetime import datetime
from multiagent.scenario import BaseScenario
import numpy as np

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

# global golbalTime 

# now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
sourceDir = "/Users/Patrick Wilk/Documents/RL/JJ_first_Look/SURPL-V2-JJ_first_look/JJ_cust"
fileOut = open(sourceDir + "/results/PPO.txt", "w+")
# fileOut = open(sourceDir + "/results/" + now + "/PPO", "w+")


# ------------------------------
#       SMART BUILDING
# ------------------------------
class MultiAgentSBEnv(gym.Env):
    def __init__(self):
        self.viewer = None
        self.load = []
        self.demand = []
        # self.demandCharge = 2 * self.load
        self.demanCharge = 0
        self.timeStep = 0
        self.deltaUtilization = 0
        self.penalty = 0
        self.totalReward = 0
        self.action_space = None
        self.observation_space = None
        self.reset()
        
    def reset(self):
        # self.load = 0
        # MOVE TO 24 HOURS FOR INITAL RESULTS NOW THAT PRELIM RESULTS ESTABLISHED
        self.demand = [3, 1, 1]
        self.load = []
        self.timeWindow = 2
        # self.timeWindow = len(self.demand) - 1
        self.timeStep = 0
        self.totalReward = 0
        # self.deltaUtilization = abs(self.demand[self.timeStep] - self.load[self.timeStep])
        self.deltaUtilization = 0
        self.penalty = (self.deltaUtilization) ** 2
        self.action_space = Box(low=0, high=self.demand[self.timeStep], shape=(1,), dtype=float)
        # OBSERVATION SPACE CAN ALSO INCLUDE DEMAND CHARGE, TIME, ETC -> 1 INSUFFICIENT FOR MORE RESULTS
        # self.observation_space = Box(low=np.array([0, 0]), high=np.array([self.deltaUtilization, self.timeWindow]), shape=(2,), dtype=float)
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([self.deltaUtilization, self.demand[self.timeStep], self.timeWindow]), shape=(3,), dtype=float)
        # return np.array(0)
        return np.array([0, 0, 0])
        # pass
        # return state
    
    def step(self, action):
        info = {}
        reward = 0
        done:bool = False
        # action[0] = abs(action[0])
        self.load.append(action[0])
        self.demandCharge = max(self.load)


        self.deltaUtilization = abs(self.demand[self.timeStep] - self.load[self.timeStep])
        self.penalty = (self.deltaUtilization) ** 2
        
        observation = self.demand[self.timeStep]
        

        
        # STILL NEED PEAK DEMAND IN REWARD
        # HOW DO WE CONTRIBUTE TO THIS REWARD?
               
        if self.timeStep == self.timeWindow:
            reward -= self.load[self.timeStep] + self.penalty + 2*(self.demandCharge)
        # IF HAVEN'T REACHED END OF SIMULATION YET
        elif self.timeStep < self.timeWindow and self.load[self.timeStep] != 0:
            reward -= self.load[self.timeStep] + self.penalty
            
        else:
            reward -= self.load[self.timeStep] + 1 + self.penalty
            
        # REWARD CAN BE DIFFERENCE BETWEEN COSTS

        self.timeStep += 1
        self.totalReward += reward
        done = True if self.timeStep > 2 else done
            
        # return np.array([0, observation]), reward, done, info
        return np.array([self.deltaUtilization, observation, self.timeWindow]), reward, done, info
    
    def render(self, mode="human"):
        screen_h = 600
        screen_w = 1000
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_w, screen_h, "SB")
            
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None