import numpy as np
import matplotlib.pyplot as plt
import math
import copy


class QLearning:
    def __init__(self, env, seed=2022):
        np.random.seed(seed)
        self.env = env
        self.best_run = (None, [], -math.inf)

    """Returns the best possible action"""
    def max_action(self, q, state, actions):
        values = np.array(q[state, a] for a in actions)
        return actions[np.argmax(values)]

    """Key component of Q-Learning"""
    def training(self, epochs=50000, alpha=0.1, gamma=1.0, epsilon=1.0, save=False, load=False, plot=True,
                 print_interval=50):

        if load:
            q = self.load_q("QMatrix")
        else:
            q = {}
        for state in self.env.state_space:
            for action in self.env.available_actions:
                q[state, action] = 0

        self.env.reset()
        total_rewards = np.zeros(epochs)
        pr = True
        for i in range(epochs):
            if not (i % print_interval):
                print("start game: ", i)
            done = False
            epoch_reward = 0
            observation = self.env.reset()
            best_env = copy.copy(self.env)
            actions = []
            while not done:
                rand = np.random.random()
                action = self.max_action(q, observation, self.env.possible_actions()) if rand < (1 - epsilon) \
                    else self.env.action_space_sample()
                next_observation, reward, done, info = self.env.step(action)
                actions.append(action)
                epoch_reward += reward
                next_action = self.max_action(q, next_observation, self.env.possible_actions())
                q[observation, action] = q[observation, action] + alpha * \
                                         (reward + gamma * q[next_observation, next_action] - q[observation, action])
                observation = next_observation

            total_rewards[i] = epoch_reward
            if epoch_reward > self.best_run[2]:
                self.best_run = (best_env, actions, epoch_reward)
        if plot:
            plt.plot(total_rewards)
            plt.show()
        if save:
            self.save_q(q, "QMatrix")


    def print_q(self, q):
        for i in range(self.env.state_space):
            for j in range(len(self.env.available_actions)):
                print(q[i, j], " ", end='')
            print("\n")


    def save_q(self, q, file_name):
        q_matrix = np.zeros((len(self.env.state_space), len(self.env.available_actions)))
        y = 0
        for state in self.env.state_space:
            x = 0
            for action in self.env.available_actions:
                q_matrix[y][x] = q[state, action]
                x += 1
            y += 1
        np.savetxt(file_name, q_matrix, delimiter=" ")

    def load_q(self, file_name):
        q = {}
        for state in self.env.state_space:
            for action in self.env.available_actions:
                q[state, action] = 0

        with open(file_name) as fn:
            q_matrix = np.loadtxt(fn, delimiter=" ")
        y = 0
        for state in self.env.state_space:
            x = 0
            for action in self.env.available_actions:
                q[state, action] = q_matrix[y][x]
                x += 1
            y += 1
        return q

    """Simulate a chain of actions in a given environment"""
    def execute(self, run):
        env_exe = run[0]
        actions_exe = run[1]
        total_reward = 0
        for action in actions_exe:
            print("Agent: ", env_exe.agent_position)
            print(env_exe.grid)
            input("Press enter for the next step")
            observation, reward, done, info = env_exe.step(action)
            total_reward += reward
            print("action: ", action," reward: ", reward, " total reward: ", total_reward)