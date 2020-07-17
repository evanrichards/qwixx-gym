import gym
import qwixx_gym
import numpy
numpy.random.seed(10)

env = gym.make("qwixx-v0")

state = env.reset()
env.render()
env.step([0, 3])
env.render()
env.step([0, 0])
env.render()
env.step([0, 0])
env.render()
