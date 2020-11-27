import gym
import qwixx_gym
import numpy
numpy.random.seed(10)

env = gym.make("qwixx-normalized-v0")

state = env.reset()
# print(list(map(lambda a: (a[0] + 1, a[1]),enumerate(env.obs.state))))
# env.render()
# take g
# env.render()
# whites 7 as red
env.step(1)
# whites 4 as green
env.step(3)
# whites 9 as red
env.step(1)
# skip
env.step(0)
# whites 10 as red
env.step(1)
# strike
env.step(0)
#whites 9 blue
env.step(4)

env.render()
# env.render()
# env.step(0)
# print(env.obs.get_dice_values())
# env.step(0)
# print(env.obs.get_dice_values())
# env.step(0)
# print(env.obs.get_dice_values())
# env.step(0)
# print(env.obs.get_dice_values())
# env.step(3)
# env.render()