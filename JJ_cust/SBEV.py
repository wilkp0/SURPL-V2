from gym import Env
from gym.spaces import Discrete, Box
from gym.envs.classic_control import rendering
from datetime import datetime
# global golbalTime 

# now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
sourceDir = "/Users/jordan/ThesisMARL/SURPL-V2/JJ_cust"
fileOut = open(sourceDir + "/results/PPO.txt", "w+")
# fileOut = open(sourceDir + "/results/" + now + "/PPO", "w+")

# ------------------------------
#       SMART BUILDING
# ------------------------------
class SmartBuildingEnv(Env):
    def __init__(self):
        self.viewer = None
        self.demand = []
        self.load = 0
        self.timeStep = 0
        # self.time = 0
        # self.reward = 0
        self.deltaUtilization = 0
        self.penalty = 0
        self.action_space = None
        self.observation_space = None
        self.reset()
        
    def reset(self):
        # self.time = 0
        self.load = 0
        self.demand = [3, 1, 1]
        # self.timeStep +=1 == self.demand[self.timeStep]
        self.timeStep = 0
        self.deltaUtilization = abs(self.demand[self.timeStep] - self.load)
        self.penalty = (self.deltaUtilization) ** 2
        self.action_space = Box(low=0, high=max(self.demand), shape=(1,), dtype=float)
        self.observation_space = Box(low=0, high=max([self.deltaUtilization]), shape=(1,), dtype=float)
        return 0 
        # pass
        # return state
    
    def step(self, action):
        info = {}
        reward = 0
        done:bool = False
        self.load = action[0]
        
        print("Stepping action: ", round(action[0], 2))
        print("*"*25)
        print("*"*25)
        print("Stepping action: ", round(action[0], 2), file=fileOut)
        print("*"*25, file=fileOut)
        print("*"*25, file=fileOut)
        
        
        # self.timeStep += 1
        while not done:
            print("Timestep: ", self.timeStep)
            print("-"*5)
            print("Load: ", round(self.load, 2))
            print("Demand: ", self.demand[self.timeStep])
            print("Timestep: ", self.timeStep, file=fileOut)
            print("-"*5, file=fileOut)
            print("Load: ", round(self.load, 2), file=fileOut)
            print("Demand: ", self.demand[self.timeStep], file=fileOut)

            # print(action)
            # self.deltaUtilization = 0
            self.deltaUtilization = abs(self.demand[self.timeStep] - self.load)
            self.penalty = (self.deltaUtilization) ** 2
            
            # self.load = action[0]
            observation = self.demand[self.timeStep]
            print("Penalty: ", round(self.penalty, 2))
            print("Delta ", round(self.deltaUtilization, 2))
            print("Observation: ", observation)
            print("-"*25)
            print("Penalty: ", round(self.penalty, 2), file=fileOut)
            print("Delta ", round(self.deltaUtilization, 2), file=fileOut)
            print("Observation: ", observation, file=fileOut)
            print("-"*25, file=fileOut)
            
            reward -= self.load + self.penalty
            self.timeStep += 1
            done = True if self.timeStep > 2 else done
            
        
        return observation, reward, done, info
    
    def render(self, mode="human"):
        screen_h = 600
        screen_w = 1000
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_w, screen_h)
            
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# ------------------------------
#            EV 
# ------------------------------
class ChargingStationEnv(Env):
    def __init__(self):
        self.viewer = None
        self.load = 0
        self.required = []
        self.timeStep = 0
        # self.time = 0
        # self.reward = 0
        self.action_space = None
        self.observation_space = None
        self.reset()
        
    def reset(self):
        # self.time = 0
        self.timeStep = 0
        self.required = [1, 1, 0]
        self.load = max(sum([l for l in self.required]), 0)
        self.action_space = Box(low=0, high=self.load, shape=(1,), dtype=float)
        self.observation_space = Box(low=0, high=self.required[self.timeStep], shape=(1,), dtype=float)
        
        return 0
        # pass
        # return state
    
    def step(self, action):
        info = {}
        reward = 0
        done: bool
        
        print("EV Stepping... ", self.timeStep)
        print("EV Stepping... ", self.timeStep, file=fileOut)
        
        done is True if self.timeStep > 2 else done is False
        
        self.timeStep += 1
        while not done:
            
            self.load = action
            observation = self.required[self.timeStep] 
            
            # reward -= (self.required) ** 2
            reward -= self.load
            
        
        return observation, reward, done, info
    
    def render(self, mode="human"):
        screen_h = 600
        screen_w = 400
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_w, screen_h)
            
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None