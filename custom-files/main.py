from stable_baselines3 import PPO
from environments import Smart_Building_Environment 
from environments import Charging_Station_Env 

def test(model_SBE, model_CSE):
    print("Testing models...")
    pass

def main():
    
    batch_size = 100
    ep_per_batch = 100

    print("Loading agents...")
    SBE = Smart_Building_Environment()
    CSE = Charging_Station_Env()

    print("Loading models...")
    model_SBE = PPO('MlpPolicy', SBE, verbose = 1)
    model_CSE = PPO('MlpPolicy', CSE, verbose = 1)

    print("Aggregating models...")
    models = [model_SBE, model_CSE]

    print("Iterating...")
    for i in range(batch_size):
        for model in models:
            model.learn(total_timesteps=ep_per_batch)
        test(*models)

if __name__ == "__main__":
   main()

