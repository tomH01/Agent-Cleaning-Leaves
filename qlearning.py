import numpy as np


class QLearning:
    def __init__(self, env, seed=2022):
        np.random.seed(seed)
        self.env = env

    """Returns the best possible action"""
    def max_action(self, q, state, possible_actions):
        values = np.array(q[state, a] for a in possible_actions)
        return possible_actions[np.argmax(values)]

    """Key component of Q-Learning"""
    def training(self, epochs=50000, alpha=0.1, gamma=1.0, epsilon=1.0, plot=True, print_interval=50):
        Q = {}
        for state in self.env.state_space:
            for action in self.env.possible_actions:
                Q[state, action] = 0

        self.env.reset()
        total_rewards = np.zeros(epochs)
        for i in range(epochs):
            if not (i % print_interval):
                print("start game: ", i)
            done = False
            epoch_reward = 0
            observation = self.env.reset()
            while not done:
                rand = np.random.random()
                action = self.max_action(Q, observation, self.env.possible_actions) if rand < (1 - epsilon) \
                    else self.env.action_space_sample()
                next_observation, reward, done, info = self.env.step(action)
                epoch_reward += reward
                next_action = self.max_action(Q, next_observation, self.env.possible_actions)
                Q[observation, action] = Q[observation, action] + alpha * \
                                         (reward + gamma * Q[next_observation, next_action] - Q[observation, action])
                observation = next_observation

            if epsilon - 2 / epochs > 0:
                epsilon -= 2 / epochs
            else:
                epsilon = 0
            total_rewards[i] = epoch_reward

        if plot:
            pass

    def print_q(self, q):
        pass

    def save_q(self, q, file_name):
        pass

    def load_q(self, file_name):
        pass



    def execute(self):
        self.env.reset
        steps = 0
        total_reward = 0
        done = False
        "To be done"
        while not done and steps < 50:
            pass

    def move(self):
        self.env.reset()
        pass
