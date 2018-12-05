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
        self.is_conv = False
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

class Conv_DQN(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size):
        super(Conv_DQN, self).__init__()
        self.is_conv = True
        self.layers = []
        self.convs = [input_shape[0], 64, 64, 128]
        for i in np.arange(len(self.convs) - 1):
            conv = nn.Conv2d(self.convs[i], self.convs[i+1], kernel_size=(3, 3), padding=1).cuda()
            nn.init.kaiming_uniform_(conv.weight)
            self.layers.extend([conv, nn.ReLU(inplace=True)])
            if i == 2:
                self.layers.extend([nn.BatchNorm2d(self.convs[-1]).cuda(), nn.MaxPool2d(2)]) # TODO: remove batch norms
        self.layers.extend([nn.BatchNorm2d(self.convs[-1]).cuda(), nn.MaxPool2d(2)])
        self.conv = nn.Sequential(*self.layers)

        self.fc = nn.Sequential(
            nn.Linear(self.convs[-1], hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)).cuda()

    def forward(self, x):
        out = self.conv(x.view(1, 2, 6, 7))
        out = out.view(out.shape[0], -1)
        return self.fc(out).view(-1)

class Transition:
    def __init__(self, state, action, reward, next_state, terminal):
        self.state = torch.FloatTensor(state.ravel()).cuda()
        self.action = action
        self.reward = torch.tensor(reward).float().cuda()
        self.lambda_return = None
        self.next_state = torch.FloatTensor(state.ravel()).cuda()
        self.terminal = terminal

class Replay_Memory:
    def __init__(self, memory_capacity):
        self.data = [None] * memory_capacity
        self.curr_index = 0
        self.length = 0

    def __len__(self):
        return self.length

    def step(self, index, amount):
        return (index + amount) % self.length

    def record(self, state, action, reward, next_state, terminal):
        transition = Transition(state, action, reward, next_state, terminal)
        self.add_transition(transition)

    def add_buffer(self, buffer):
        for transition in buffer:
            self.add_transition(transition)

    def add_transition(self, transition):
        self.data[self.curr_index] = transition
        self.curr_index = (self.curr_index + 1) % len(self.data)
        self.length = min(self.length + 1, len(self.data))

    def refresh_returns(self, player, discount, future_weight, only_new=False):
        if self.length == 0: return

        start = self.step(self.curr_index, -1)
        i = start
        while True:
            curr = self.data[i]
            if curr.lambda_return is not None and only_new: break

            curr.lambda_return = curr.reward
            if not self.data[i].terminal:
                next = self.data[self.step(i, 1)]
                with torch.no_grad():
                    curr.lambda_return = curr.lambda_return + discount * \
                        (future_weight * next.lambda_return + \
                        (1 - future_weight) * torch.max(player.q_net(next.state)))

            i = self.step(i, -1)
            if i == start: break

    def sample(self, batch_size):
        populated = self.data[:self.length]
        return np.random.choice(populated, min(len(populated), batch_size), replace=False)

class Agent:
    def __init__(self, game, memory_capacity, player_num, q_net=None):
        self.replay_memory = Replay_Memory(memory_capacity)
        self.q_net = q_net
        if q_net is None:
            self.q_net = DQN(np.prod(game.input_size), 512, game.output_size).cuda()
            #self.q_net = Conv_DQN(game.conv_input_shape, 256, game.output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters())
        #self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=1e-5)

        self.player_num = player_num
        self.is_human = False

    def compute_move(self, game):
        return self.choose_action(game)

    # epsilon greedy for now, try other strategies later
    def choose_action(self, game, random_prob=0):
        valid_moves = game.board.valid_moves()
        if random.random() >= random_prob:
            input = torch.FloatTensor(game.features(conv=True)).cuda()
            with torch.no_grad():
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

def restart_if_needed(game, players, active):
    if game.turn is None:
        player1_result = game_result(game.winner, 1)
        game.reset()
        if game.turn != players[active].player_num:
            players[1 - active].act(game)
        return player1_result
    return None

# simultaneously train player 1 and 2 to play game
def train(game, players=None):
    memory_capacity = 10000
    if players is None:
        players = [Agent(game, memory_capacity, 1), Agent(game, memory_capacity, -1)]
    win_rates = {'random': [0.5, 0.5], 'mostly_random': [0.5, 0.5]}
    metrics = {'random': behaviors.Random(), 'mostly_random': behaviors.MostlyRandom()}
    player1_win_rate = 0.5

    discount = 0.99
    random_prob = 0.6 # epsilon in epsilon-greedy
    random_prob_decay = 0.985
    random_prob_floor = 0.02
    batch_size = 8
    epochs = 1000
    iterations = 200
    future_weight = 0.6 # how much to weight future returns vs q-estimate (lambda in the literature)
    refresh_interval = 4 # for eligibility traces

    params = [next(players[0].q_net.parameters()).data.clone(), next(players[1].q_net.parameters()).data.clone()]
    for i in range(epochs):
        active = i % 2
        print('Epoch {} (player {})'.format(i, active + 1))
        losses, rewards = [], []
        history_len = min(i + 1, 25)

        if i % refresh_interval == 0:
            players[0].replay_memory.refresh_returns(players[0], discount, future_weight)
            players[1].replay_memory.refresh_returns(players[1], discount, future_weight)

        memory_buffer = [] # TODO: handle case where this doesn't get added to replay_memory because epoch ended before game did

        curr_players = list(players)
        # if win_rates['random'][active] < 0.8:
        #     curr_players[1 - active] = metrics['random']
        #     print('   Training with Random')
        # elif win_rates['mostly_random'][active] < 0.8:
        #     curr_players[1 - active] = metrics['mostly_random']
        #     print('   Training with Mostly Random')

        if i % 25 == 0:
            torch.save(players[0].q_net, 'player1_{}.pt'.format(i))
            torch.save(players[1].q_net, 'player2_{}.pt'.format(i))
            print('Saving...')

        if game.turn == players[1 - active].player_num:
            curr_players[1 - active].act(game)
            result = restart_if_needed(game, curr_players, active)
            if result is not None and curr_players[1 - active] == players[1 - active]:
                player1_win_rate = moving_avg(player1_win_rate, result, history_len)

        for t in range(iterations):
            state = game.features(conv=players[active].q_net.is_conv)
            action = players[active].act(game, random_prob)

            if game.turn == players[1 - active].player_num:
                curr_players[1 - active].act(game)

            reward = game.reward(players[active].player_num)
            rewards.append(reward)
            next_state = game.features(conv=players[active].q_net.is_conv)

            memory_buffer.append(Transition(state, action, reward, next_state, game.turn is None))

            result = restart_if_needed(game, curr_players, active)
            if result is not None:
                if curr_players[1 - active] == players[1 - active]:
                    player1_win_rate = moving_avg(player1_win_rate, result, history_len)
                players[active].replay_memory.add_buffer(memory_buffer)
                players[active].replay_memory.refresh_returns(players[active], discount, future_weight, only_new=True)
                memory_buffer = []

            if len(players[active].replay_memory) > memory_capacity // 10:
                for transition in players[active].replay_memory.sample(batch_size):
                    players[active].optimizer.zero_grad()
                    target = transition.lambda_return
                    predicted = autograd.Variable(players[active].q_net(transition.state)[transition.action], requires_grad=True)
                    loss = players[active].criterion(predicted, target)
                    losses.append(loss.item())
                    loss.backward()
                    players[active].optimizer.step()

        if random_prob > random_prob_floor:
            random_prob *= random_prob_decay

        print('   Avg Loss: {}'.format(np.mean(losses)))
        print('   Avg Reward: {}'.format(np.mean(rewards)))

        simulation_players = list(players)
        for i in range(2):
            for alg_name in metrics:
                simulation_players[1 - active] = metrics[alg_name]
                winner = game.simulate(simulation_players)
                win_rates[alg_name][active] = moving_avg(win_rates[alg_name][active], game_result(winner, players[active].player_num), history_len)

        print('   Avg Win Rate (vs Random): {}'.format(win_rates['random'][active]))
        print('   Avg Win Rate (vs Mostly Random): {}'.format(win_rates['mostly_random'][active]))
        print('   Avg Win Rate (vs Player 2): {}'.format(player1_win_rate))
        print('')

    return players

import connect4
def main():
    players = None
    while True:
        players = train(connect4.ConnectFour(), players)
        gui = connect4.GUI(connect4.ConnectFour(), players[0], behaviors.Human())
        gui.mainloop()
        gui = connect4.GUI(connect4.ConnectFour(), players[0], players[1])
        gui.mainloop()

if __name__ == '__main__':
    main()
