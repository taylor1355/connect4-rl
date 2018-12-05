import numpy as np
from tkinter import Tk, Button, Frame, Canvas, font
import behaviors
import time

class ConnectFour:
    def __init__(self, board=None):
        self.board = board
        self.reset()

        # for neural networks
        self.input_size = self.board.data.size * 2
        self.conv_input_shape = (2,) + self.board.data.shape
        self.output_size = self.board.cols

    def move(self, col):
        if self.turn != None:
            self.board.move(col, self.turn)
            self.moves += 1
            winner = self.board.winner()
            if winner is None and self.moves < self.board.data.size:
                self.turn = -self.turn
            else:
                self.turn = None
                self.winner = winner

    def reward(self, player):
        if player == self.winner:
            return 1
        elif -player == self.winner:
            return -1
        return 0

    def reset(self):
        if self.board is None: self.board = ScalableBoard()
        else: self.board.reset()
        self.turn = 1 # 1 is player 1, -1 is player 2
        self.moves = 0
        self.winner = None

    def features(self, conv=False):
        if conv:
            features = np.empty(self.conv_input_shape)
            features[0] = self.board.data > 0
            features[1] = self.board.data < 0
            return features.astype(float)
        flattened = self.board.data.ravel()
        return np.concatenate((flattened > 0, flattened < 0)).astype(float)

    def simulate(self, players):
        new_game = ConnectFour()
        player_indices = {1:0, -1:1}
        while new_game.turn is not None:
            new_game.move(players[player_indices[new_game.turn]].compute_move(new_game))
        return new_game.winner

class ScalableBoard:
    ZERO = np.array([0, 0], dtype=int)
    UP, RIGHT = np.array([1, 0], dtype=int), np.array([0, 1], dtype=int)
    DOWN, LEFT = -UP, -RIGHT

    def __init__(self):
        self.rows, self.cols = 6, 7
        self.data = np.empty((self.rows, self.cols), dtype=int)
        self.top_rows = np.empty(self.cols, dtype=int)
        self.reset()

    def move(self, col, player):
        if self.top_rows[col] < self.rows:
            pos = np.array([self.top_rows[col], col], dtype=int)
            self.data[tuple(pos)] = player
            self.last_played = pos
            self.top_rows[col] += 1

    def valid_moves(self):
        return np.where(self.top_rows < self.rows)[0]

    def winner(self):
        if self.last_played is None: return None

        win = (self.win_from_pos(self.last_played, self.UP) or
            self.win_from_pos(self.last_played, self.RIGHT) or
            self.win_from_pos(self.last_played, self.UP + self.RIGHT) or
            self.win_from_pos(self.last_played, self.UP - self.RIGHT))

        if win: return self.data[tuple(self.last_played)];
        return None

    def win_from_pos(self, pos, dir):
        count = 1
        for sign in [-1, 1]:
            curr_pos = pos + dir * sign
            while self.in_bounds(curr_pos) and self.data[tuple(pos)] == self.data[tuple(curr_pos)]:
                curr_pos += dir * sign
                count += 1
        return count >= 4

    def in_bounds(self, pos):
        return np.sum(pos < self.ZERO) + np.sum(pos >= self.data.shape) == 0

    def reset(self):
        self.data.fill(0)
        self.top_rows.fill(0)
        self.last_played = None

class GUI:
    def __init__(self, game, player1, player2):
        self.app = Tk()
        self.app.title('Connect4')
        self.app.resizable(width=False, height=False)
        self.players = {1: player1, -1: player2}
        self.game = game
        self.board = game.board
        self.buttons = {}
        self.frame = Frame(self.app, borderwidth=1, relief="raised")
        self.tiles = {}

        self.human_turns = set()
        if player1.is_human:
            self.human_turns.add(1)
        if player2.is_human:
            self.human_turns.add(-1)

        # init buttons
        for x in range(self.board.cols):
            handler = lambda x=x: self.human_move(x)
            button = Button(self.app, command=handler, font=font.Font(family="Helvetica", size=14), text=x+1)
            button.grid(row=0, column=x, sticky="WE")
            self.buttons[x] = button

        # init tiles
        self.frame.grid(row=1, column=0, columnspan=self.board.cols)
        for row, col in np.ndindex(self.board.data.shape):
            tile = Canvas(self.frame, width=60, height=50, bg="navy", highlightthickness=0)
            tile.grid(row=self.board.rows-1-row, column=col)
            self.tiles[row, col] = tile

        # handler = lambda: self.reset()
        # self.restart = Button(self.app, command=handler, text='reset')
        # self.restart.grid(row=2, column=0, columnspan=self.board.width, sticky="WE")
        self.update()
        self.ai_move()

    # def reset(self):
    #     self.board = Board()
    #     self.update()

    def human_move(self, col):
        if self.game.turn in self.human_turns:
            self.app.config(cursor="watch")
            self.app.update()
            self.game.move(col)
            self.update()
            # move = self.board.best()
            # if move!=None:
            #     self.board = self.board.move(move)
            #     self.update()
            self.app.config(cursor="")
            self.ai_move()

    def ai_move(self):
        while self.game.turn not in self.human_turns and self.game.turn is not None:
            action = self.players[self.game.turn].compute_move(self.game)
            self.app.update()
            time.sleep(1)
            self.game.move(action)
            self.update()

    def update(self):
        for row, col in np.ndindex(self.board.data.shape):
            player = self.board.data[row, col]
            if player == 0: # empty
                self.tiles[row, col].create_oval(10, 5, 50, 45, fill="black", outline="blue", width=1)
            if player == 1: # player 1
                self.tiles[row, col].create_oval(10, 5, 50, 45, fill="yellow", outline="blue", width=1)
            if player == -1: # player 2
                self.tiles[row, col].create_oval(10, 5, 50, 45, fill="red", outline="blue", width=1)
        for col in range(self.board.cols):
            if self.board.data[-1, col] == 0:
                self.buttons[col]['state'] = 'normal'
            else:
                self.buttons[col]['state'] = 'disabled'
        winning = self.game.winner != None
        if winning:
            # for x,y in winning:
            #     self.tiles[x,y].create_oval(25, 20, 35, 30, fill="black")
            for col in range(self.board.cols):
                self.buttons[col]['state'] = 'disabled'

    def mainloop(self):
        self.app.mainloop()

if __name__ == '__main__':
    game = ConnectFour()
    player2 = behaviors.Human()
    player1 = behaviors.MostlyRandom()

    gui = GUI(game, player1, player2)
    gui.mainloop()
