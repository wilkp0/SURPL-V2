from stable_baselines3 import PPO
from environments import SmartBuilding 
from environments import ChargingStation

# def test(model_SBE, model_CSE):
def test(**models):
    print("Testing models...")
    model_SBE, modelCSE = models[0], models[1]
    # print(models[0])
    pass

def main():
    
    batch_size = 3
    ep_per_batch = 3
    print("Loading agents...")
    SBE = SmartBuilding()
    CSE = ChargingStation()

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

