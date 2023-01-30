from grid import Grid
from qlearning import QLearning


def main():
    env = Grid(dimension=(15, 15), p_leaves=0.6, reward_conf=(-1, 5, -5, 100), max_moves=200, state_mode="nbh")
    ql = QLearning(env)

    ql.training(epochs=500, alpha=0.01, gamma=0.8, epsilon=0.7 , print_interval=100, save=True)
    ql.execute(ql.best_run)


if __name__ == "__main__":
    main()

