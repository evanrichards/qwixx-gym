from gym.envs.registration import register

register(
    id='qwixx-v0',
    entry_point='qwixx_gym.envs:QwixxOneHotEnv',
)

register(
    id='qwixx-simple-v0',
    entry_point='qwixx_gym.envs:QwixxSimple',
)

register(
    id='qwixx-flat-v0',
    entry_point='qwixx_gym.envs:QwixxFlat',
)

register(
    id='qwixx-normalized-v0',
    entry_point='qwixx_gym.envs:QwixxNormalized',
)