import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete


class MultiAgentEnv(gym.Env):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            #CONTINOUS ACTION (BOX)
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                #CONTINOUS FROM -U_RANGE TO U_RANGE    
                #SHAPE = dim_p
                #float32
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
                # NOT SURE
            if agent.smart:
                total_action_space.append(u_action_space)
                # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                #WHAT ARE YOU COMMUNICATING WITH LOW 0 AND HIGH 1
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
                #IF AGENT COMMUNICATES APPEND C TO ACTION_SPACE
            if not agent.silent:
                total_action_space.append(c_action_space)
                # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    #THIS IS FOR CONTINOUS, BECOMES A TUPLE
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            # OBSERVATION SPACE IS BOX FROM -INF TO INF
            # SHAPE = obs_dim
            #FLOAT23
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            # OBSERVATION C = dim_c
            agent.action.c = np.zeros(self.world.dim_c)




    def step(self, action_n):
        obs_n = []
        reward_n = []
        self.load.append(self.action[0])
        self.demandCharge = max(self.load)
        #done:bool = False
        done_n = []
        info_n = {'n': []}
        #self.load.append(action_n[0])
        #self.demandCharge = max(self.load)
        #self.deltaUtilization = abs(self.demand[self.timeStep] - self.load[self.timeStep])
        #self.penalty = (self.deltaUtilization) ** 2

        #observation = self.demand[self.timeStep]
        #if self.timeStep == self.timeWindow:
            #reward -= self.load[self.timeStep] + self.penalty + 2*(self.demandCharge)
    # IF HAVEN'T REACHED END OF SIMULATION YET
        #elif self.timeStep < self.timeWindow and self.load[self.timeStep] != 0:
            #reward -= self.load[self.timeStep] + self.penalty
            
        #else:
            #reward -= self.load[self.timeStep] + 1 + self.penalty

        #self.timeStep += 1
        #self.totalReward += reward
        #done = True if self.timeStep > 2 else done

        #if done:

        #return np.array([observation, self.timeWindow]), reward, done, info

        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
        
        self.timeStep +=1 
        done_n = True if self.timeStep > 2 else False 

        return obs_n, reward_n, done_n, info_n


    def reset(self):

        self.demand = [3, 1, 1]
        self.timeWindow = 2
        self.timeStep = 0
        self.totalReward = 0
        self.deltaUtilization = 0
        self.penalty = (self.deltaUtilization) ** 2
        #self.action_space = Box(low=0, high=self.demand[self.timeStep], shape=(1,), dtype=float)
        #self.observation_space = Box(low=np.array([0, 0]), high=np.array([self.demand[self.timeStep], self.timeWindow]), shape=(2,), dtype=float)
        # return np.array(0)
        #return np.array([0, 0])

        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)


        # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # USED !! unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        done = True if self.timeStep > 2 else done
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)


    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        #OUR ACTIONS ARE CONTINOUS(NOT USED)
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
            else:
                agent.action.u = action[0]

            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0        


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n