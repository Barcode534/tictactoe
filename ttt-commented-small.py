import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model
from keras.utils import plot_model
from keras import optimizers
import random
import numpy as np
import math

reward_dep = .7
x_train = True

model = Sequential()
model.add(
    Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(units=130, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

plot_model(model, to_file='model.png', show_shapes= True, show_layer_names=True)

model_2 = Sequential()
model_2.add(
    Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(units=130, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.add(Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


def one_hot(state):
    current_state = []

    for square in state: # state is a list of the squares, square is an indiv square. 0 = empty. 1 = X. -1 = O.
        if square == 0:
            current_state.append(1)
            current_state.append(0)
            current_state.append(0)
        elif square == 1:
            current_state.append(0)
            current_state.append(1)
            current_state.append(0)
        elif square == -1:
            current_state.append(0)
            current_state.append(0)
            current_state.append(1) #this code converts a 3x3 grid with values, (or a list of 9 locations as it's treated here) into a list of 27 values.
                                    # empty = 000, X = 010, O = 001.

    return current_state


def get_outcome(state):
    total_reward = 0

    for i in range(0, 9):
        if i == 0 or i == 3 or i == 6: # start in a left hand cell, if the follow 2 cells match, reward = 1 or -1 or 0.
            if state[i] == state[i + 1] and state[i] == state[i + 2]:
                total_reward = state[i]
                break
            elif state[0] == state[4] and state[0] == state[8] and i == 0: #dia line top left corner to bottom right.
                total_reward = state[0]
                break
        if i < 3:
            if state[i] == state[i + 3] and state[i] == state[i + 6]:
                total_reward = state[i]
                break
            elif state[2] == state[4] and state[2] == state[6] and i == 2:
                total_reward = state[2]
                break

    if (state[0] == state[1] == state[2]) and not state[0] == 0: ##superceding if statements for the 8 possible lines.
        total_reward = state[0]
    elif (state[3] == state[4] == state[5]) and not state[3] == 0:
        total_reward = state[3]
    elif (state[6] == state[7] == state[8]) and not state[6] == 0:
        total_reward = state[6]
    elif (state[0] == state[3] == state[6]) and not state[0] == 0:
        total_reward = state[0]
    elif (state[1] == state[4] == state[7]) and not state[1] == 0:
        total_reward = state[1]
    elif (state[2] == state[5] == state[8]) and not state[2] == 0:
        total_reward = state[2]
    elif (state[0] == state[4] == state[8]) and not state[0] == 0:
        total_reward = state[0]
    elif (state[2] == state[4] == state[6]) and not state[2] == 0:
        total_reward = state[2]

    return total_reward


try:
    model = load_model('tic_tac_toe.h5')
    model_2 = load_model('tic_tac_toe_2.h5')
    print('Pre-existing model found... loading data.')
except:
    pass


def process_games(games, model, model_2):
    global x_train # taken from the x_train outside of the function
    xt = 0 # initialise X, O and drawn wins
    ot = 0
    dt = 0
    states = []
    q_values = []
    states_2 = []
    q_values_2 = []

    for game in games: #game is the full game history from all empty squares, to the finished game state. games is a list of all these game histories.
        total_reward = get_outcome(game[len(game) - 1]) # minus 1 because length is magnitude starting at 1 whereas information starts at 0. So this line gets the final result of the game
        if total_reward == -1: ## these lines add to the sum of scores for this set of 2000 games
            ot += 1
        elif total_reward == 1:
            xt += 1
        else:
            dt += 1
        # print('------------------')
        # print(game[len(game) - 1][0], game[len(game) - 1][1], game[len(game) - 1][2])
        # print(game[len(game) - 1][3], game[len(game) - 1][4], game[len(game) - 1][5])
        # print(game[len(game) - 1][6], game[len(game) - 1][7], game[len(game) - 1][8])
        # print('reward =', total_reward)

        for i in range(0, len(game) - 1):
            if i % 2 == 0: #this shows if it was X's turn???
                for j in range(0, 9):
                    if not game[i][j] == game[i + 1][j]: ##i is the gamestate history leading up to the final gamestate. j is the individual squares in the game histories. so this is comparing individual squares and seeing if they changed
                        reward_vector = np.zeros(9) # return an array of a given shape filled with zeros.
                        reward_vector[j] = total_reward * (reward_dep ** (math.floor((len(game) - i) / 2) - 1)) # this exponent returns values between 0 and 4, based on game length. if you're in the won gamestate, total reward == reward vector == 0 or 1 or -1
                        # the further you are away from finishing the game, the reward_dep(th?) factor diminishes the reward value of a winning state. if a states becomes winning for X, but you are on move 1, making that move has a low value. Making the final winning move is high value.
                        # print(reward_vector)
                        states.append(game[i].copy()) ##add each specific game state to the states list.
                        q_values.append(reward_vector.copy()) # add each reward vector to the q values list.
                        #print(game[i])
                        #print(reward_vector)
            else:
                for j in range(0, 9):
                    if not game[i][j] == game[i + 1][j]: ##copy for O's(?)
                        reward_vector = np.zeros(9)
                        reward_vector[j] = -1 * total_reward * (reward_dep ** (math.floor((len(game) - i) / 2) - 1))
                        # print(reward_vector)
                        states_2.append(game[i].copy())
                        q_values_2.append(reward_vector.copy())
                        #print(game[i])
                        #print(reward_vector)

    if x_train:
        zipped = list(zip(states, q_values))
        random.shuffle(zipped)
        states, q_values = zip(*zipped) #the '*' is used to unzip the list.
        new_states = []
        for state in states: #this takes a random game state, converts it into binary 27 length list, and adds to new state
            new_states.append(one_hot(state))

        # for i in range(0, len(states)):
        # print(new_states[i], states[i], q_values[i])
        # print(np.asarray(new_states))

        model.fit(np.asarray(new_states), np.asarray(q_values), epochs=4, batch_size=len(q_values), verbose=1)
        model.save('tic_tac_toe.h5')
        del model
        model = load_model('tic_tac_toe.h5')
        print(xt / 20, ot / 20, dt / 20) #these are %s of Xwin, O win, draw, in training.
    else:
        zipped = list(zip(states_2, q_values_2))
        random.shuffle(zipped)
        states_2, q_values_2 = zip(*zipped)
        new_states = []
        for state in states_2:
            new_states.append(one_hot(state))

        # for i in range(0, len(states)):
        # print(new_states[i], states[i], q_values[i])
        # print(np.asarray(new_states))

        model_2.fit(np.asarray(new_states), np.asarray(q_values_2), epochs=4, batch_size=len(q_values_2), verbose=1)
        model_2.save('tic_tac_toe_2.h5')
        del model_2
        model_2 = load_model('tic_tac_toe_2.h5')
        print(xt / 20, ot / 20, dt / 20)

    x_train = not x_train #this basically just alternates x_train from True to False to True.

# win = 1; draw = 0; loss = -1 --> moves not taken are 0 in q vector


mode = input('Choose a mode: (training/playing) ')

while True: #this is basically the main function inside a while true loop
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # sides --> 0 = Os, 1 = Xs
    games = []
    current_game = []

    if mode == 'training':
        print(x_train)
        # total_games = int(input('How many games should be played? '))
        total_games = 2000
        # e_greedy = float(input('What will the epsilon-greedy value be? '))
        e_greedy = .7

        for i in range(0, total_games):
            playing = True
            nn_turn = True
            c = 0 #used later for choosing a random square
            board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            # sides --> 0 = Os, 1 = Xs
            current_game = []
            current_game.append(board.copy())
            nn_board = board

            while playing: #playing will remain true until the game ends. hence we stick in here.
                if nn_turn:
                    if random.uniform(0, 1) <= e_greedy: #if less than epsil_greedy value, pick random square
                        choosing = True
                        while choosing:
                            c = random.randint(0, 8)
                            if board[c] == 0: #if square is empty, add 1.
                                choosing = False
                                board[c] = 1
                                current_game.append(board.copy())
                            # save state to game array
                    else: #else take model prediction. Interesting as it's random 70% of the time. Explains why so many wins in training.
                        pre = model.predict(np.asarray([one_hot(board)]), batch_size=1)[0] #converts current board state into the 27 length binary game state.
                                        # [0] because the model comes out like [[]] so we want to remove one set of square brackets.
                        highest = -1000
                        num = -1
                        for j in range(0, 9):
                            if board[j] == 0: #if square is empty, check prediction for that move. keep highest scoring move.
                                if pre[j] > highest:
                                    highest = pre[j].copy()
                                    num = j

                        choosing = False
                        board[num] = 1
                        current_game.append(board.copy())

                else:
                    if random.uniform(0, 1) <= e_greedy:
                        choosing = True
                        while choosing:
                            c = random.randint(0, 8)
                            if board[c] == 0:
                                choosing = False
                                board[c] = -1
                                current_game.append(board.copy())
                            # save state to game array
                    else:
                        pre = model_2.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
                        highest = -1000
                        num = -1
                        for j in range(0, 9):
                            if board[j] == 0:
                                if pre[j] > highest:
                                    highest = pre[j].copy()
                                    num = j

                        choosing = False
                        board[num] = -1
                        current_game.append(board.copy())

                playable = False

                for square in board:
                    if square == 0:
                        playable = True #if there's available squares, continue playing
                # elif find square and check

                if not get_outcome(board) == 0:
                    playable = False #unless there is a winner, 1 or -1.

                # print(get_outcome(board))

                if not playable:
                    playing = False #this would be a tie, i believe. all squares full, and no winner.

                nn_turn = not nn_turn #flip turn over. True = X turn. False = O turn

            # print(board[0], board[1], board[2])
            # print(board[3], board[4], board[5])
            # print(board[6], board[7], board[8])

            games.append(current_game) #add this game to games
        # print('current game:', current_game)

        process_games(games, model, model_2)
    elif mode == 'playing':
        print('')
        print('A new game is starting!')
        print('')

        team = input('Choose a side: (x/o) ')
        print('')

        board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        running = True
        x_turn = True
        while running:
            if (x_turn and team == 'o') or (not x_turn and not team == 'o'): ##if it's x turn and human is O, AI move. if O turn and team is X, AI move.
                if team == 'o':
                    pre = model.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
                elif team == 'x':
                    pre = model_2.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
                # print(pre)
                print('')
                highest = -1000
                num = -1
                for j in range(0, 9):
                    if board[j] == 0:
                        if pre[j] > highest:
                            highest = pre[j].copy()
                            num = j

                print(pre)

                # TODO: ADD EXTRA IF STATEMENT FOR NUM == -1 (FIRST OPTION ALWAYS TRUMPS)

                if team == 'o':
                    board[num] = 1
                elif team == 'x':
                    board[num] = -1
                x_turn = not x_turn
                print('AI is thinking...')
            else:
                move = int(input('Input your move: ')) ##make human move
                if board[move] == 0:
                    if team == 'o':
                        board[move] = -1
                    elif team == 'x':
                        board[move] = 1
                    x_turn = not x_turn
                else:
                    print('Invalid move!')

            r_board = []

            for square in board: #board for printing
                if square == 0:
                    r_board.append('-')
                elif square == 1:
                    r_board.append('x')
                elif square == -1:
                    r_board.append('o')

            print(r_board[0], r_board[1], r_board[2])
            print(r_board[3], r_board[4], r_board[5])
            print(r_board[6], r_board[7], r_board[8])

            full = True

            for square in board:
                if square == 0:
                    full = False

            if full:
                running = False
                if get_outcome(board) == 0:
                    print('The game was drawn!')

            if not get_outcome(board) == 0:
                running = False
                print(get_outcome(board), 'won the game!')