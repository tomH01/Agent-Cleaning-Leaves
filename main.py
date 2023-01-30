from grid import Grid
from qlearning import QLearning


def main():
    env = Grid(dimension=(15, 15), p_leaves=0.6, reward_conf=(-1, 5, -1, 100), max_moves=200, state_mode="nbh")
    ql = QLearning(env)

    ql.training(epochs=2000, alpha=0.01, gamma=0.8, epsilon=0.7 , print_interval=100)


if __name__ == "__main__":
    main()

