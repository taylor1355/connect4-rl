from __future__ import division

from collections import deque
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import copy

import behaviors

# to select action, use exploration policy from
# epsilon-greedy, uct equation, sample from action distribution given by Q
# try fully connected and convolutional representation

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size))
        self.network.apply(self.init_weights)

    def init_weights(self, module):
        if type(module) == nn.Linear:
            torch.nn.init.xavier_uniform(module.weight)

    def forward(self, x):
        return self.network(x)

class Sample:
    def __init__(self, state, action, reward, next_state, terminal):
        self.state = autograd.Variable(torch.FloatTensor(state.ravel()), requires_grad=True).cuda()
        self.action = action
        self.reward = autograd.Variable(torch.tensor(reward).float(), requires_grad=True).cuda()
        self.next_state = autograd.Variable(torch.FloatTensor(state.ravel()), requires_grad=True).cuda()
        self.terminal = terminal

class Replay_Memory:
    def __init__(self, memory_capacity):
        self.data = [None] * memory_capacity
        self.curr_index = 0
        self.length = 0

    def __len__(self):
        return self.length

    def record(self, state, action, reward, next_state, terminal):
        sample = Sample(state, action, reward, next_state, terminal)
        self.data[self.curr_index] = sample
        self.curr_index = (self.curr_index + 1) % len(self.data)
        self.length = min(self.length + 1, len(self.data))

    def sample(self, batch_size):
        populated = self.data[:self.length]
        return np.random.choice(populated, min(len(populated), batch_size), replace=False)

class Agent:
    def __init__(self, game, memory_capacity):
        self.replay_memory = Replay_Memory(memory_capacity)
        self.q_net = DQN(np.prod(game.input_size), 256, game.output_size).cuda()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=5e-4)
        #self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=1e-5)

        self.is_human = False

    def compute_move(self, game):
        return self.choose_action(game)

    # epsilon greedy for now, try other strategies later
    def choose_action(self, game, random_prob=0):
        valid_moves = game.board.valid_moves()
        if random.random() >= random_prob:
            input = torch.FloatTensor(game.features()).cuda()
            action = torch.argmax(self.q_net(input)).item()
            if action in valid_moves:
                return action
        return np.random.choice(valid_moves)

    def act(self, game, random_prob=0):
        action = self.choose_action(game, random_prob)
        game.move(action)
        return action

def game_result(winner, player):
    if winner == player: return 1
    elif winner == player: return 0
    else: return 0.5

def moving_avg(curr, update, history_len):
    return (history_len - 1) / history_len * curr + 1 / history_len * update

# simultaneously train player 1 and 2 to play game
def train(game, players=None):
    memory_capacity = 10000
    if players is None:
        players = [Agent(game, memory_capacity), Agent(game, memory_capacity)]
    target_nets = [copy.deepcopy(player.q_net) for player in players]
    player_nums = [1, -1]
    win_rates = {'random': [0.5, 0.5], 'mostly_random': [0.5, 0.5]}
    metrics = {'random': behaviors.Random(), 'mostly_random': behaviors.MostlyRandom()}
    player1_win_rate = 0.5

    discount = 0.99
    random_prob = 0.5
    random_prob_decay = 0.975
    random_prob_floor = 0.01
    batch_size = 8
    epochs = 150
    iterations = 200

    params = [next(players[0].q_net.parameters()).data.clone(), next(players[1].q_net.parameters()).data.clone()]
    for i in range(epochs):
        print('Epoch {}'.format(i))
        active = i % 2
        losses, rewards = [], []
        history_len = min(i + 1, 25)
        for t in range(iterations):
            if game.turn == player_nums[1 - active]:
                players[1 - active].act(game)
                if game.turn is None:
                    player1_win_rate = moving_avg(player1_win_rate, game_result(game.winner, 1), history_len)
                    game.reset()
                    continue

            state = game.features()
            action = players[active].act(game, random_prob)

            if game.turn == player_nums[1 - active]:
                players[1 - active].act(game)

            reward = game.reward(player_nums[active])
            rewards.append(reward)
            next_state = game.features()
            players[active].replay_memory.record(state, action, reward, next_state, game.turn is None)

            if game.turn is None:
                player1_win_rate = moving_avg(player1_win_rate, game_result(game.winner, 1), history_len)
                game.reset()

            curr_batch_size = 0 if len(players[active].replay_memory) < memory_capacity // 10 else batch_size
            for sample in players[active].replay_memory.sample(curr_batch_size):
                players[active].optimizer.zero_grad()
                target = sample.reward
                if not sample.terminal:
                    target = target + discount * torch.max(target_nets[active](sample.next_state))
                predicted = players[active].q_net(sample.state)[sample.action]
                loss = players[active].criterion(predicted, target)
                losses.append(loss.item())
                loss.backward()
                players[active].optimizer.step()

        target_nets[active] = copy.deepcopy(players[active].q_net)

        if random_prob > random_prob_floor:
            random_prob *= random_prob_decay

        print('   Avg Loss: {}'.format(np.mean(losses)))
        print('   Avg Reward: {}'.format(np.mean(rewards)))

        curr = next(players[active].q_net.parameters()).data.clone()
        if params[active] is None: params[active] = curr
        print('   Change: {}'.format(torch.sum(torch.abs(curr - params[active]))))
        params[active] = curr

        simulation_players = list(players)
        for alg_name in metrics:
            simulation_players[1 - active] = metrics[alg_name]
            winner = game.simulate(simulation_players)
            win_rates[alg_name][active] = moving_avg(win_rates[alg_name][active], game_result(winner, player_nums[active]), history_len)

        print('   Avg Win Rate (vs Random): {}'.format(win_rates['random'][active]))
        print('   Avg Win Rate (vs Mostly Random): {}'.format(win_rates['mostly_random'][active]))
        print('   Avg Win Rate (vs Player 2): {}'.format(player1_win_rate))
        print('')

    return players

import connect4
players = None
while True:
    players = train(connect4.ConnectFour(), players)
    gui = connect4.GUI(connect4.ConnectFour(), players[0], behaviors.Human())
    gui.mainloop()
    gui = connect4.GUI(connect4.ConnectFour(), players[0], players[1])
    gui.mainloop()
