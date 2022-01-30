import gym 
import custom_env

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

env.close()