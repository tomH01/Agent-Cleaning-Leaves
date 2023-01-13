class Grid:
    def __init__(self, dimension, p_leaves, rand_w, reward_conf, max_moves=350):
        self.max_moves = max_moves
        self.dim = dimension
        self.p_leaves = p_leaves
        self.rand_w = rand_w
        self.reward_conf = reward_conf

        self.actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.possible_actions = ['U', 'D', 'L', 'R']

        self.agent_position = (1, 1)
        self.goal = (self.dimension[0]-1, self.dimension[1]-1)


    def init_grid(self, dimension):
        m = dimension[0]
        n = dimension[1]
