from dll_agent import DQNAgent
import gym
import os
import time
import numpy
import qwixx_gym
from qwixx_gym.envs.qwixx_one_hot_env import WHITE_ACTION_COLOR, COLOR_ACTION

# numpy.random.seed(10)

# Potential variables
max_episode = 10000
mini_batch_size = 10
print_every = 10
save_every = 100
sample_every = 1000
path = "/Users/richards/Desktop/qwixx_model"


def sample(sagent, senv):
    sis_done = False
    sstate = senv.reset()
    while not sis_done:
        print("Turn begin ===================", sagent.epsilon)
        senv.render()
        saction = sagent.act(sstate, True)
        print("white action", WHITE_ACTION_COLOR[saction[0]])
        print("color action", COLOR_ACTION[saction[1]])
        snext_state, sreward, sis_done, snotes = senv.step(saction)
        print(snotes)
        sstate = snext_state
    senv.render()


def run_training():
    env = gym.make("qwixx-v0")
    state = env.reset()
    agent = DQNAgent(path, 0.99, env.action_space, gamma=0.65,
                     state_size=66, action_size=2)
    agent.save()
    episode = 0
    now = time.time()
    game_lengths = []
    turns = 0
    print("{:>5s}: {:>15s}{:>10s}{:>10s}".format("Run", "Elapsed time", "Max score", "% Error"))
    episode_high_score = []
    errors = []
    while True:
        turns += 1
        action = agent.act(state)
        # print("white action", WHITE_ACTION_COLOR[action[0]])
        # print("color action", COLOR_ACTION[action[1]])
        next_state, reward, is_done, notes = env.step(action)
        if not agent.last_action_was_random:
            print(reward, is_done, notes)
        agent.remember(state, action, reward, next_state, is_done)
        state = next_state
        if len(agent.memory) > mini_batch_size:
            agent.replay(mini_batch_size)
        if is_done:
            if episode % save_every == 0 and episode != 0:
                agent.save()
            episode_high_score.append(max(notes.get("scores", [0])))
            errors.append(1 if "error" in notes else 0)
            if episode % print_every == 0 and episode != 0:
                print("{:5d}:{:15d}s{:>10d}{:>10d}%".
                      format(episode, int(time.time() - now), max(episode_high_score),
                             int((sum(errors) / print_every) * 100)))
                episode_high_score = []
                errors = []
            if episode % sample_every == 0 and episode != 0:
                sample(agent, env)
            episode += 1
            env.reset()
            game_lengths.append(turns)
            turns = 0

    print("Played {} games in {}s with an average of {} turns".
          format(max_episode, int(time.time() - now), sum(game_lengths) // len(game_lengths)))
    agent.save()


if __name__ == "__main__":
    run_training()
