from time import time
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2 #idk yet
        num_agents = 2
        world.num_agents = num_agents
        num_landmarks = 0
        num_adversaries = 0

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.adversary = True if i < num_adversaries else False

        # add landmarks
        #No landmarks in our simulation 
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):

        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        #benchmark data is the everchanging distance 
        # this can be used in EV for the remaing charge
        # not sure if needed in the SB since there is a penalty instead
        if agent.adversary:
            return 0
        else:
            self.demandCharge = max(agent.load)
            time = []
            time.append(self.timeStep)
            return tuple(time)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    
    #define Smart buildign Reward 
    
    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        SB_reward = True
        SB_adv_reward = False

        # Calculate reward for adversary
        adversary_agents = self.adversaries(world)
        if SB_adv_reward:  # distance-based adversary reward
            adv_rew = sum([(a.self.load - a.self.demand) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0

#??????????????????????????????????????????????????????????????????????????????????
        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if SB_reward:  # distance-based agent reward
            if self.timeStep == self.timeWindow:
                pos_rew -= ([self.load[self.timeStep] + self.penalty + 2*(self.demandCharge) for a in good_agents])
            elif self.timeStep < self.timeWindow and self.load[self.timeStep] != 0:
                pos_rew -= ([self.load[self.timeStep] + self.penalty for a in good_agents])
            else: 
                pos_rew -= ([self.load[self.timeStep] + 1 + self.penalty for a in good_agents])

        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            
        return pos_rew + adv_rew



    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        SB_reward = True
        if SB_reward:  # distance-based reward
            return sum(agent.self.load - agent.self.demand) 
        else:  # proximity-based reward (binary)
            adv_rew = 0
            return adv_rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        good_agents = self.good_agents(world)

        deltaUtilization = []
        for agent in good_agents:
            deltaUtilization.append.abs(self.demand[self.timeStep] - self.load[self.timeStep])
     
        observation = []
        for agent in good_agents:
            observation.append.self.demand[self.timeStep]
        
        self.timeWindow
        return np.concatenate([agent.abs(self.demand[self.timeStep] - self.load[self.timeStep]) + agent.self.demand[self.timeStep] + self.timeWindow])
       