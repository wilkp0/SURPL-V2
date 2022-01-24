from gym import Env 
from gym.spaces import Discrete, Box 
import numpy as np
import random 

class ShowerEnv(Env):
    def __init__(self):
        #Actions we can take: Down, stay, up 
        self.action_space= Discrete(3)
        #Temperature array
        self.observation_space = Box(low=np.array([0], high=np.array([100])))
        #Set start temp
        self.state = 38 + random.ranint(-3,3)
        #Set shower length  (episodes)
        self.shower_length = 60
    def step(self,action):
        #0-1 = -1 temp
        #1-1 = 0 nothing
        #2-1 = 1 temp
        self.state += action-1
        #decrease shower length by 1 second 
        self.shower_length -= 1

        #Calculate Reward 
        if self.state >=37 and self.state <=39:
            reward = 1
        else:
            reward = -1
        if self.shower_length <=0:
            done = True
        else: 
            done = False 
        
        self.state += random.randomint(-1,1)

        # placeholder for info required by OpenAI
        info={}

        return self.state, reward, done, info 

        pass
    def render(self):
        #implement visualization 
        pass
    def reset(self):
        self.state= 38 + random.randomint(-3,3)
        #Reset shower time 
        self.shower_length= 60 
        return self.state 

        ## creates custom environment 
env= ShowerEnv()
        ## will give examples of results within the action space 
env.action_space.sample()
        ## will give examples of observation within observation spave
env.observation_space.sample()

#runs through 10 different showers 
episodes = 10 
for episodes in range (1, episodes + 1):
    state = env.reset()
    done= False 
    score = 0 

    while not done:
        env.render()
        #takes a sample action [0,1,2]
        action = env.action_space.sample()
        #apply action to the environment
        n_state, reward, done, info = env.step(action)
        score += reward 

states = env.observation_space.shape
actions = env.action_space.n


def build_model(states, actions):
    model.Sequential()
    model.add(Dense(24, activation= 'relu', input_shape=states))
    model.add(Dense(24, activation= 'relu'))
    model.add(Dense(actions, activation='linear'))
    return model 

model = build_model(states, actions)

model.summary()





       
