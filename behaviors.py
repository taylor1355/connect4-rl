from collections import deque
import numpy as np
import copy

class Human:
    def __init__(self):
        self.is_human = True

class Random:
    def __init__(self):
        self.is_human = False

    def compute_move(self, game):
        return np.random.choice(game.board.valid_moves())

    def act(self, game):
        action = self.compute_move(game)
        game.move(action)
        return action

class MostlyRandom:
    def __init__(self, ):
        self.is_human = False

    def compute_move(self, game):
        player = game.turn
        moves = game.board.valid_moves()
        scores = [np.exp(self.score(game, player, move) * 3) for move in moves]
        move = np.random.choice(moves, p=scores/np.sum(scores))
        return move

    def act(self, game):
        action = self.compute_move(game)
        game.move(action)
        return action

    def score(self, game, player, move):
        simulation = copy.deepcopy(game)
        simulation.move(move)
        if simulation.turn is None: return simulation.reward(player) * 1.1
        worst_reward = None
        moves = simulation.board.valid_moves()
        for i in range(len(moves)):
            if i > 0:
                simulation = copy.deepcopy(game)
                simulation.move(move)
            simulation.move(moves[i])
            if worst_reward is None or simulation.reward(player) < worst_reward:
                worst_reward =  simulation.reward(player)
        return worst_reward
