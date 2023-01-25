import numpy as np
class QLearning:
    def __init(self, env, seed=2022):
        np.random.seed(seed)
        self.env = env

    """Returns the best possible action"""
    def max_action(self, q, state, possible_actions):
        values = np.array(q[state, a] for a in possible_actions)
        return possible_actions[np.argmax(values)]

    """Key component of Q-Learning"""
    def training(self):
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

