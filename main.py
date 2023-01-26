from grid import Grid
from qlearning import QLearning


def main():
    env = Grid(dimension=(10, 10), p_leaves=0.1)
    ql = QLearning(env)

    ql.training(epochs=5000, alpha=0.1, gamma=1.0, epsilon=1.0)


if __name__ == "__main__":
    main()

