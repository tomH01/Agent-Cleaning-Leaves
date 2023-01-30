import copy
import math
import random as rand
import numpy as np


class Grid:
    def __init__(self, dimension, p_leaves, rand_w=3, p_wind=0.5, reward_conf=(-1, 5, -1, 5), max_moves=350, state_mode="nbh"):
        self.count = 0
        self.max_moves = max_moves
        self.dimension = dimension
        self.p_leaves = p_leaves
        self.rand_w = rand_w
        self.p_wind = p_wind
        self.w = rand_w
        self.reward_conf = reward_conf
        self.leaves = []
        self.collected_leaves = 0
        self.grid = copy.deepcopy(self.init_grid())
        self.state_mode = state_mode
        match self.state_mode:
            case "nbh":
                self.state_space = [i for i in range(512)]
            case "pos":
                self.state_space = [i for i in range(pow(2, int(math.log(dimension[0], 2))
                                                         + int(math.log(dimension[1], 2)) + 2))]
            case "nbh + pos":
                self.state_space = [i for i in range(pow(2, 9 + int(math.log(dimension[0], 2))
                                                         + int(math.log(dimension[1], 2)) + 2))]
        self.actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1), 'S': (0, 0)}
        self.available_actions = ['U', 'D', 'L', 'R', 'S']
        self.agent_position = (0, 0)
        self.goal = (self.dimension[0]-1, self.dimension[1]-1)
        self.moves = 0



    """Initializes the grid with a percentage of randomly placed leaves. 
    Air is represented with '0' and leaves with '1' """
    def init_grid(self):
        m = self.dimension[0]
        n = self.dimension[1]
        num_leaves = round(m * n * self.p_leaves)
        self.leaves = []
        k = np.zeros((m, n))
        count = 0
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
                    self.leaves.remove(lv)
                    self.leaves.append((new_y_pos, lv[1]))

    def is_goal(self, position):
        return position[0] == self.goal[0] and position[1] == self.goal[1]

    def update_pos(self, new_position):
        self.agent_position = new_position

    def reset(self):
        self.grid = copy.deepcopy(self.init_grid())
        self.moves = 0
        self.collected_leaves = 0
        self.w = self.rand_w
        self.update_pos((0, 0))
        return self.agent_state()

    def action_space_sample(self):
        return np.random.choice(self.possible_actions())

    def step(self, action):
        y_pos = self.agent_position[0] + self.actions[action][0]
        x_pos = self.agent_position[1] + self.actions[action][1]
        self.update_pos((y_pos, x_pos))
        self.moves += 1
        self.w -= 1
        reward = 0
        if self.moves == self.max_moves:
            return self.agent_state(), reward, True, {}

        if self.is_goal(self.agent_position):
            reward = self.reward_conf[3] * self.collected_leaves
        elif self.grid[y_pos][x_pos] == 1 and action == 'S':
            self.grid[y_pos][x_pos] = 0
            self.leaves.remove((y_pos, x_pos))
            self.collected_leaves += 1
            reward = self.reward_conf[1]
        elif self.grid[y_pos][x_pos] == 0 and action == 'S':
            reward = self.reward_conf[2]
        else:
            reward = self.reward_conf[0]

        if self.w == 0:
            self.w = self.rand_w
            self.move_leaves()
        return self.agent_state(), reward, self.is_goal(self.agent_position), {}

    def possible_actions(self):
        actions = ['S']
        neighborhood = self.neighborhood()
        for i in range(9):
            if neighborhood[i] != 5:
                match i:
                    case 1:
                        actions.append('U')
                    case 3:
                        actions.append('L')
                    case 5:
                        actions.append('R')
                    case 7:
                        actions.append('D')
        return actions

    def neighborhood(self):
        y_pos = self.agent_position[0]
        x_pos = self.agent_position[1]
        neighborhood = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                y_state = y_pos + i
                x_state = x_pos + j
                if 0 <= y_state < self.dimension[0] and 0 <= x_state < self.dimension[1]:
                    neighborhood.append(self.grid[y_state][x_state])
                else:
                    neighborhood.append(5)

        return neighborhood

    def pos_to_bin(self):
        n_bin = [int(x) for x in list('{0:0b}'.format(self.agent_position[0]))]
        m_bin = [int(x) for x in list('{0:0b}'.format(self.agent_position[1]))]
        n = [0] * (int(math.log(self.dimension[0], 2) + 1 - len(n_bin))) + n_bin
        m = [0] * (int(math.log(self.dimension[1], 2) + 1 - len(m_bin))) + m_bin
        return n + m

    def agent_state(self):
        state = []
        nbh = self.neighborhood()
        pos = self.pos_to_bin()
        match self.state_mode:
            case "nbh":
                state = nbh
            case "pos":
                state = pos
            case "nbh + pos":
                state = nbh + pos

        int_state = 0
        key = 0
        for cell in state:
            if cell == 1:
                int_state += pow(2, key)
            key += 1
        return int_state
