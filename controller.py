import connect4
import behaviors
import time

def main():
    players = {1: behaviors.Random(), -1: behaviors.Random()}
    game = connect4.ConnectFour()

    start = time.time()
    num_games = 1000

    for i in range(num_games):
        while game.turn is not None:
            game.move(players[game.turn].compute_move(game))
        game.reset()

    total = time.time() - start
    per_game = total / num_games
    print('{} games played in {} sec ({} ms / game)'.format(num_games, total, per_game * 1000))

if __name__ == '__main__':
    main()
