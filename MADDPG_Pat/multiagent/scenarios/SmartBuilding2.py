import numpy as np
from multiagent.coreNew import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        num_adversaries = 0
        #commuincation
        num_agents = 2
        world.num_agents = num_agents
        self.demand = [3,1,1]

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            #COMMUNICAION
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        # random properties for landmarks
        # set goal landmark
        # set random initial states
        self.demand = [3, 1, 1]
        self.load = []
        self.timeWindow = 2

        self.timeStep = 0
        self.totalReward = 0

        self.deltaUtilization = 0
        self.penalty = (self.deltaUtilization) ** 2 



    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        # our benchmark data is load ? and remaining required when doing EV?
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        SB_reward = True 
        EV_reward = False

        # NOT USED AS OF NOW Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        for a in adversary_agents:
            adv_rew = 0
       
        # Calculate NEGATIVE reward for agents
        good_agents = self.good_agents(world)

        if SB_reward:  # distance-based agent reward OUR CASE IF SB

            if self.timeStep == self.timeWindow:
                pos_rew = -([a.load[self.timeStep] + a.penalty + 2*(a.demandCharge) for a in good_agents])
            # IF HAVEN'T REACHED END OF SIMULATION YET
            elif self.timeStep < self.timeWindow and self.load[self.timeStep] != 0:
                pos_rew = -([a.load[self.timeStep] + a.penalty for a in good_agents])
            
            else:
                pos_rew = -([a.load[self.timeStep] + 1 + a.penalty for a in good_agents])
        return pos_rew + adv_rew

    #NOT USED FOR THIS SIMULATION LEAVE UNCHANGED
    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        SB_reward = True
        if SB_reward:  # distance-based reward
            return -np.sum(0)
        else:  # proximity-based reward (binary)
            adv_rew = 0
            return adv_rew        

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        good_agents = self.good_agents(world)


        observation = []        
        for a in good_agents:
            observation.append(a.load)

        self.timeWindow = [] 
        self.timeWindow.append(self.timeWindow)

        return np.concatenate( self.timeWindow)
    