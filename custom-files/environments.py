'''
File for the independent version of the environment
Ideally, the code would be housed in the same place
but that seems unlikely considering how the different policies interact
'''

from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from gym import spaces, Env, Wrapper

class ChargingStation(Env):
    def __init__(self):
        # super().__init_()
        # spaces.Discrete(3)
        self.reset()
        self.time = 3
        # self.action_space = spaces.Discrete(3)
        self.action_space = Box(low=0, high=self.required, shape=(1,), dtype=float)
        self.observation_space = Box(low=0, high=25, shape=(1,), dtype=float)
        pass

    def reward(self):
        reward = 0
        reward -= (self.required**2)
        reward -= (self.load**2)
        
        return reward

    def done(self):
        if self.time >= 2:
            return True
        else:
            return False

    def step(self, action):
        self.time_step += 1
        self.load = 0
        self.load = action
        self.required -= self.load
        self.required = max(self.required, 0)

        observation = self.load
        reward = self.reward()
        done = self.done()
        
        # observation, reward, done = self.Env.step(action)
        
        return observation, reward, done, {}
    
    def _take_action(self, action):
        pass

    def reset(self):
        self.time_step = 0
        self.required = 2
        self.load = 0
        return 0
    
    def render(self, mode="human", close=False):
        pass


class SmartBuilding(Env):
    def __init__(self):
        # super().__init_()
        self.reset()
        self.demand = [1, 1, 0]
        self.time = 3
        # self.action_space = spaces.Discrete(3)
        self.action_space = Box(0, max(self.demand), (1,), float)
        self.observation_space = Box(0, 25, (1,), float)
        pass

    def reward(self):
        reward = 0
        reward -= abs(self.load - self.demands[self.time_step])**2
        reward -= (self.load**2)
        
        return reward

    def done(self):
        if self.time >= 2:
            return True
        else:
            return False

    def step(self, action):
        self.time_step += 1
        self.load = 0
        self.load = action
        observation = self.load
        reward = self.reward()
        done = self.done()
        
        return observation, reward, done, {}

    def reset(self):
        self.time_step = 0
        self.demands = [3, 1, 1]
        # self.demands = [1.5, .5, .5]
        self.load = 0   
        return 0
    
    def render(self, mode="human", close=False):
        pass


class dual_environment(Env):
    def __init__(self) -> None:
        super().__init_()
        self.reset()
        self.action_space = Box(0, max(self.demand), (2,), float)
        self.observation_space = Box(0, 25, (2,), float)
        pass

    def reward(self):
        reward = 0
        reward -= abs(self.load - self.demands[self.time_step])**2
        reward -= (self.load**2)
        
        return reward

    def done(self):
        if self.time >= 2:
            return True
        else:
            return False

    def step(self, actions):
        self.time_step += 1
        self.load = 0
        self.load = actions[0] + actions[1]
        observation = self.load
        reward = self.reward()
        done = self.done()
        
        return observation, reward, done, {}

    def reset(self):
        self.time_step = 0
        self.demands = [3, 1, 1]
        self.load = 0   
        return 0
    
    def render(self, mode="human", close=False):
        pass