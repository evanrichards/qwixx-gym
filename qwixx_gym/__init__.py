from gym.envs.registration import register

register(
    id='qwixx-v0',
    entry_point='qwixx_gym.envs:QwixxOneHotEnv',
)