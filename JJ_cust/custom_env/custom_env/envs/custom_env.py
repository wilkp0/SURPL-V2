from gym import Env
from gym.spaces import Discrete, Box
from gym.envs.classic_control import rendering
# global golbalTime 

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
        self.timeStep = 0
        self.deltaUtilization = abs(self.demand[self.timeStep] - self.load)
        self.penalty = (self.deltaUtilization) ** 2
        self.action_space = Box(low=0, high=max(self.demand), shape=(1,), dtype=float)
        self.observation_space = Box(low=0, high=max([self.deltaUtilization]), shape=(1,), dtype=float)
        # return 0 
        # pass
        # return state
    
    def step(self, action):
        info = {}
        reward = 0
        done:bool = False
        
        print("SB Stepping... ", self.timeStep)
        
        done is True if self.timeStep > 2 else done is False
        
        self.timeStep += 1
        while not done:
            # self.deltaUtilization = 0
            self.deltaUtilization = abs(self.demand[self.timeStep] - self.load)
            self.penalty = (self.deltaUtilization) ** 2
            
            self.load = action
            observation = self.demand[self.timeStep]
            
            reward -= self.load + self.penalty
        
        
        return observation, reward, done, info
    
    def render(self):
        screen_h = 600
        screen_w = 400
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_w, screen_h)
    
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
        self.load = max(sum([l for l in self.required[self.timeStep]])[0], 0)
        self.action_space = Box(low=0, high=self.load, shape=(1,), dtype=float)
        self.observation_space = Box(low=0, high=self.required[self.timeStep], shape=(1,), dtype=float)
        
        
        pass
        # return state
    
    def step(self, action):
        info = {}
        reward = 0
        done: bool
        
        print("EV Stepping... ", self.timeStep)
        
        done is True if self.timeStep > 2 else done is False
        
        self.timeStep += 1
        while not done:
            
            self.load = action
            observation = self.required[self.timeStep] 
            
            # reward -= (self.required) ** 2
            reward -= self.load
            
        
        return observation, reward, done, info
    
    def render(self):
        screen_h = 600
        screen_w = 400
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_w, screen_h)
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None