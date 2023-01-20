import copy
import random as rand
import numpy as np


class Grid:
    def __init__(self, dimension, p_leaves, rand_w=3, p_wind=0.5, reward_conf=(-1, 5, -1, 5), max_moves=350):
        self.dimension = dimension
        self.max_moves = max_moves
        self.p_leaves = p_leaves
        self.dim = dimension
        self.rand_w = rand_w
        self.p_wind = p_wind
        self.w = rand_w
        self.reward_conf = reward_conf
        self.leaves = []
        self.collected_leaves = 0

        self.grid = copy.deepcopy(self.init_grid())

        self.actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1), 'S': (0, 0)}
        self.possible_actions = ['U', 'D', 'L', 'R', 'S']

        self.agent_position = (1, 1)
        self.goal = (self.dimension[0]-1, self.dimension[1]-1)

        self.moves = 0

    def init_grid(self):
        m = self.dimension[0]
        n = self.dimension[1]
        num_leaves = round(m * n * self.p_leaves)
        k = np.zeros((m, n))
        while len(self.leaves) < num_leaves:
            rand_pos = (rand.randint(0, m-1), rand.randint(0, n-1))
            if k[rand_pos] == 0:
                k[rand_pos] = 1
                self.leaves.append(rand_pos)
        return k

    """Moves leaves UP with probability p_wind"""
    def move_leaves(self):
        for lv in self.leaves:
            if rand.random() < self.p_wind:
                new_y_pos = (lv[0]-1) % self.dimension[0]
                if not (self.grid[new_y_pos][lv[1]] == 1 or self.agent_position == (new_y_pos, lv[1])):
                    self.grid[lv[0]][lv[1]] = 0
                    self.grid[new_y_pos][lv[1]] = 1

    def is_goal(self, position):
        return position[0] == self.goal[0] and position[1] == self.goal[1]

    def reset(self):
        self.moves = 0
        self.updatePos((1, 1))
        return self.agentState()

    def step(self, action):
        x_pos = self.agent_position[0] + self.actions[action][0]
        y_pos = self.agent_position[1] + self.actions[action][1]

        self.moves += 1
        self.w -= 1

        if self.moves == self.max_moves:
            pass

        if self.is_goal(self.agent_position):
            reward = self.reward_conf[3] * self.collected_leaves
        elif self.grid[y_pos][x_pos] == 1 and action == 'S':
            self.grid[y_pos][x_pos] = 0
            reward = self.reward_conf[1]
        elif self.grid[y_pos][x_pos] == 0 and action == 'S':
            reward = self.reward_conf[2]
        else:
            reward = self.reward_conf[0]

        if self.w == 0:
            self.w = self.rand_w
            self.move_leaves()

        new_state = self.agentState()
        return new_state, reward, self.is_goal(self.agent_position), {}

    def agent_state(self):
        y_pos = self.agent_position[0]
        x_pos = self.agent_position[1]
        state = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not i == j:



