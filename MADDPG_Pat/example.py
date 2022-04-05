import numpy as np
from make_env import make_env


env = make_env('simple_adversary')
print('number of agetns',env.n)
print ('observation space', env.observation_space)
print ("action space", env.action_space.shape)
#index what agetn you want 
print('n actions', env.action_space[0].n)

observation = env.reset
print(observation)

no_op = np.array([1,0.1,0,0.33,0.4])
action = [no_op, no_op, no_op]
obs_, reawrd, done, info = env.step(action)

print(reward)
print(done)



