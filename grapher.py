import matplotlib.pyplot as plt
from deep_q_learning import *
import numpy as np
import behaviors
import connect4
import torch
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python grapher.py <graph dir>")
        return
    dir = sys.argv[1]

    epochs = np.arange(0, 1000, 25)
    win_rates = {'Random': [[], []], 'Mostly Random': [[], []]}
    opponents = {'Random': behaviors.Random(), 'Mostly Random': behaviors.MostlyRandom()}
    player_nums = {0:1, 1:-1}

    game = connect4.ConnectFour()
    for epoch in epochs:
        print(epoch)
        for player in [0, 1]:
            model = torch.load(dir + '/player{}_{}.pt'.format(player + 1, epoch))
            agent = Agent(game, 1, player_nums[player], q_net=model)
            for alg_name in opponents:
                players = [agent, opponents[alg_name]]
                if player == 1: players = players[::-1]
                results = []
                for round in range(200):
                    winner = game.simulate(players)
                    results.append(game_result(winner, player_nums[player]))
                win_rates[alg_name][player].append(np.mean(results))

    colors = {'Random': ['-r', '-c'], 'Mostly Random': ['-k', '-b']}
    for player in [0, 1]:
        for alg_name in opponents:
            plt.plot(epochs, win_rates[alg_name][player], colors[alg_name][player], label="Player {} vs {}".format(player + 1, alg_name))

    plt.title(sys.argv[1])
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
