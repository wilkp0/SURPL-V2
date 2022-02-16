from gym.envs.registration import register

register(id="SmartBuilding_single-v0",
         entry_point="custom_env.envs:SmartBuildingEnv"
)

register(id="ChargingStation_single-v0",
        entry_point="custom_env.envs:ChargingStationEnv"
)