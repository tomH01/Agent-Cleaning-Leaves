from grid import Grid
from qlearning import QLearning


def main():
    env = Grid(dimension=(10, 10), p_leaves=0.01, reward_conf=(-1, 0, 0, 50000), max_moves=350)
    ql = QLearning(env)

    ql.training(epochs=500, alpha=0.1, gamma=0.1, epsilon=0.8, print_interval=100)


if __name__ == "__main__":
    main()

