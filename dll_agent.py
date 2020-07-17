from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
import os
from keras.models import model_from_json


class DQNAgent:
    def __init__(self, path, epsilon_decay, action_space,
                 state_size=None, action_size=None, epsilon=1.0, epsilon_min=0.01,
                 gamma=1, alpha=.01, alpha_decay=.01, gamma_decay=1, gamma_min=0.1):
        self.memory = deque(maxlen=100000)
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.gamma_decay = gamma_decay
        self.gamma_min = gamma_min
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.path = path  # location where the model is saved to

        self.last_action_was_random = False
        self.model = self._build_model()
        self.last_ten_actions = deque(maxlen=10)

    def _build_model(self):
        # try:
        if (os.path.exists(os.path.join(self.path, "model.json"))
                and os.path.exists(os.path.join(self.path, "model.h5"))):
            json_file = open(os.path.join(self.path, "model.json"), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(os.path.join(self.path, "model.h5"))
            print("Loaded model from disk", loaded_model)

            loaded_model.compile(loss='mse',
                                 optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
            return loaded_model

        # except Exception as e:
        #     print("failed to load", e.)
        print("building model")
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model

    def act(self, state, verbose=False):
        if np.random.random() <= self.epsilon:
            self.last_action_was_random = True
            # if verbose:]
            return self.action_space.sample()
        self.last_action_was_random = False
        weights = self.model.predict(np.array([state]))
        action = np.argmax(weights)
        frequency = sum(map(lambda x: 1 if x == action else 0, self.last_ten_actions))
        if (len(self.last_ten_actions) == 10
                and frequency > 5):
            first = weights[0][action]
            weights[0][action] = -1000000
            second = np.argmax(weights)
            print("last 10 moves, {} were {}: {}. second {}: {}".format(frequency, action, first, second, weights[0][second]))
            self.last_action_was_random = True
            action = second
        else:
            print("not random")
            self.last_ten_actions.append(action)
        formatted_action = np.array([action % 5, action // 5])
        # indicides = weights[0].argsort()[-3:][::-1]
        # print(list(map(lambda x: (x, weights[0][x]), indicides)))
        # if verbose:
        #     print("weights", dict(enumerate(weights[0])))
        #     print("max", action)
        #     print("formatted", formatted_action)
        assert self.action_space.contains(formatted_action), "{} {}".format(formatted_action, action)
        return formatted_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            serial_action = 5 * action[0] + action[1]
            y_target = self.model.predict(np.array([state]))
            predict_out = self.model.predict(np.array([next_state]))[0]
            max_predict = np.max(predict_out)
            y_target[0][serial_action] = reward if done else reward + self.gamma * max_predict
            x_batch.append(state)
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        model_json = self.model.to_json()
        with open(os.path.join(self.path, "model.json"), "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(os.path.join(self.path, "model.h5"))
        print("Saved model to disk")
