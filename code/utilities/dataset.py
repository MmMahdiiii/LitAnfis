import os

import chess.pgn
import gym
import gym_chess
import numpy as np
from collections import deque


class ChessDataset:
    def __init__(self, root_path, read_size):
        #  root path indicates the path were the pgn files exists.
        #  then the procedure below finds all the png files within the root path.
        self.pgns = []
        try:
            for filename in os.listdir(root_path):
                if filename.endswith('.pgn'):
                    self.pgns.append(os.path.join(root_path, filename))
        except Exception as e:
            print(f'Unable to traverse the root path: {e}')
            #  PAY ATTENTION HERE:
            exit(0)

        self.file_index = 0

        i = 0
        while True:
            try:
                self.current_open_file = open(self.pgns[self.file_index])
                break
            except Exception as e:
                if i == len(self.pgns):
                    print(
                        f'Looks like none of the pgn files can be opened, the process needs to be terminated:{e}')
                    #  PAY ATTENTION HERE:
                    exit(0)
                print(f'Failed to open the pgn file {self.pgns[self.file_index]}: {e}\nprocedure will try to open another pgn file.')
                self.file_index = (self.file_index + 1) % (len(self.pgns))
                i += 1

        self.read_size = read_size
        self.states_pool = deque()
        self.actions_pool = deque()
        self.results_pool = deque()

        # logging purposes
        self.game_count = 0
        self.opened_files = 1
        self.invokes = 0

    def read_data(self):
        # logging:
        if self.invokes == 10:
            print(f'Opened {self.opened_files} to read {self.game_count} games in {self.invokes} invocations.')
            self.game_count = 0
            self.opened_files = 1
            self.invokes = 0

        self.invokes += 1
        self.states_pool.clear()
        self.actions_pool.clear()
        self.results_pool.clear()
        env = gym.make('ChessAlphaZero-v0')
        game = None
        while len(self.results_pool) < self.read_size:
            try:
                game = chess.pgn.read_game(self.current_open_file)
            except Exception as e:
                print(f'Could not parse the game because: {e}')
                continue

            self.game_count += 1

            if game is None:
                #  if the game was None, then all games in this png is read.
                #  and we should update our pgn file to the next file in the list of png files.
                i = 0
                while True:
                    try:
                        self.current_open_file.close()
                        self.file_index = (self.file_index + 1) % (len(self.pgns))
                        self.current_open_file = open(self.pgns[self.file_index])
                        break
                    except Exception as e:
                        if i == len(self.pgns):
                            print(
                                f'Looks like none of the pgn files can be opened, the process needs to be terminated:{e}')
                            #  PAY ATTENTION HERE:
                            exit(0)
                        print(f'Failed to open the pgn file {self.pgns[self.file_index]}: {e}\nprocedure will try to open another pgn file.')
                        i += 1
                self.opened_files += 1
                continue

            try:
                state = env.reset()
            except Exception as e:
                print(
                    f'There was a problem resetting the environment. Without environment there is nothing much to do.\n{e}')
                #  PAY ATTENTION HERE:
                exit(0)
            try:
                if game.headers['Result'] == '1-0':  # white won.
                    result = 1
                elif game.headers['Result'] == '0-1':  # black won.
                    result = -1
                else:  # draw.
                    result = 0
            except Exception as e:
                print(f'Game does not have the header "Result", try to read another game.\n{e}')
                continue

            for move in game.mainline_moves():
                try:
                    move = chess.Move.from_uci(move.uci())
                    action = env.encode(move)
                except Exception as e:
                    print(f'Illegal move found in the game, ignore this game and read another one.')
                    break

                #  the first plane after the board history positions is the color plane.
                #  14*8 = 112
                #  ATTENTION: as you can see it is assumed that the history is 8, if you've assumed another number,
                #  you should change the environment's history number as well.
                if state[0][0][112] == 1:  # it was the whites turn
                    add_result = result
                elif state[0][0][112] == 0:  # it was the blacks turn
                    add_result = -result
                self.states_pool.append(state)
                self.actions_pool.append(action)
                self.results_pool.append(add_result)
                # print(env.render(mode='unicode'))
                try:
                    state = env.step(action)[0]
                except Exception as e:
                    print(
                        f'An error occurred performing the action {env.decode(action)}, ignore this game and read another.')
                    print(f'Also the appended state and action should be deleted.')
                    self.states_pool.pop()
                    self.actions_pool.pop()
                    self.results_pool.pop()
                    print(f'The error log: {e}')
                    break

    def sample(self, batch_size):
        indices = np.random.choice(len(self.results_pool), size=batch_size, replace=False)

        states = np.array([self.states_pool[i] for i in indices], dtype=np.float32)
        actions = np.array([self.actions_pool[i] for i in indices], dtype=np.int32)
        results = np.array([self.results_pool[i] for i in indices], dtype=np.float32)

        return states, actions, results

    def full_sample(self):
        states = np.array(self.states_pool, dtype=np.float32)
        actions = np.array(self.actions_pool, dtype=np.int32)
        results = np.array(self.results_pool, dtype=np.float32)

        return states, actions, results


if __name__ == '__main__':
    env = ChessDataset('../sample_pgn', 1000)
    env.read_data()
    env.read_data()
    states, actions, results = env.sample(64)
    print(f'states: {states.shape}')
    print(f'actions: {actions.shape}')
    print(f'results: {results.shape}')
