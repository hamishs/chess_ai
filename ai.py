#chess ai class
import chess, numpy as np

class ValueFunction:
    #material score
    material_scores = {"P" : 1, "N" : 3, "B": 3, "R" : 5, "Q" : 9}

    # value of piece at board position
    value_tables = {
    "P" : np.array([
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 5, 10, 10,-20,-20, 10, 10,  5],
        [ 5, -5,-10,  0,  0,-10, -5,  5],
        [ 0,  0,  0, 20, 20,  0,  0,  0],
        [ 5,  5, 10, 25, 25, 10,  5,  5],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ]),
    "N" : np.array([
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20,   0,   5,   5,   0, -20, -40],
        [-30,   5,  10,  15,  15,  10,   5, -30],
        [-30,   0,  15,  20,  20,  15,   0, -30],
        [-30,   5,  15,  20,  20,  15,   0, -30],
        [-30,   0,  10,  15,  15,  10,   0, -30],
        [-40, -20,   0,   0,   0,   0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ]),
    "B" : np.array([
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10,   5,   0,   0,   0,   0,   5, -10],
        [-10,  10,  10,  10,  10,  10,  10, -10],
        [-10,   0,  10,  10,  10,  10,   0, -10],
        [-10,   5,   5,  10,  10,   5,   5, -10],
        [-10,   0,   5,  10,  10,   5,   0, -10],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ]),
    "R" : np.array([
        [ 0,  0,  0,  5,  5,  0,  0,  0],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [ 5, 10, 10, 10, 10, 10, 10,  5],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ]),
    "Q" : np.array([
        [-20, -10, -10, -5, -5, -10, -10, -20],
        [-10,   0,   5,  0,  0,   0,   0, -10],
        [-10,   5,   5,  5,  5,   5,   0, -10],
        [  0,   0,   5,  5,  5,   5,   0,  -5],
        [ -5,   0,   5,  5,  5,   5,   0,  -5],
        [-10,   0,   5,  5,  5,   5,   0, -10],
        [-10,   0,   0,  0,  0,   0,   0, -10],
        [-20, -10, -10, -5, -5, -10, -10, -20]
    ])
    }

    @staticmethod
    def piece_position_score(row, col, table):
        #return score of piece in position by row and col
        return table[row, col]

    @staticmethod
    def value_function(board):
        #return value of board

        score_black = 0
        score_white = 0

        #get board as fen string and evaluate piece positions
        fen = board.fen()
        fen = fen.split(' ', 1)[0]
        list_fen = list(fen)

        row = 0
        col = 0

        #analyse fen string
        for x in list_fen:
            if (x == '/'): #if / then end of row
                col = 0
                row = row + 1
            elif (x in [str(elem) for elem in range(1,9)]): #skip this many columns
                col = col + (int(x) % 8)
                #for each piece get its position and add to score
                #black
            elif (x not in ('k', 'K')):
                if (x == str.upper(x)):
                    #piece is white
                    #find value function value of piece in position
                    score_white += ValueFunction.piece_position_score(7 - row, col, ValueFunction.value_tables[x])
                    score_white += ValueFunction.material_scores[x]
                    col += 1
                else:
                    #piece is black
                    #find value function value of piece in position
                    score_black += ValueFunction.piece_position_score(row, col, ValueFunction.value_tables[str.upper(x)])
                    score_black += ValueFunction.material_scores[str.upper(x)]
                    col += 1
        return score_white - score_black
        

class AI:

    INFINITE = 999999
    
    #@staticmethod
    def minimax(board, depth, maximizing):
        if (depth == 0):
            return ValueFunction.value_function(board)

        if (maximizing):
            best_score = -AI.INFINITE
            for move in board.legal_moves:
                board_clone = board
                board_clone.push(move)

                score = AI.minimax(board_clone, depth - 1, False)
                best_score = max(best_score, score)

            return best_score
        else:
            best_score = AI.INFINITE
            for move in board.legal_moves:
                board_clone = board
                board_clone.push(move)

                score = AI.minimax(board_clone, depth - 1, True)
                best_score = min(best_score, score)

            return best_score
    
    #@staticmethod
    def alphabeta(board, depth, a, b, maximizingPlayer):
        #alpha beta search algo
        if (depth == 0):
            return ValueFunction.value_function(board)

        if (maximizingPlayer):
            best_score = -AI.INFINITE
            for move in board.legal_moves:
                board_clone = board.copy()
                board_clone.push(move)
                best_score = max(best_score, AI.alphabeta(board_clone, depth - 1, a, b, False))
                a = max(a, best_score)
                if (b<=a):
                    break
            
            return best_score
        else:
            best_score = AI.INFINITE
            for move in board.legal_moves:
                board_clone = board.copy()
                board_clone.push(move)

                best_score = min(best_score, AI.alphabeta(board_clone, depth-1, a, b, True))
                b = min(b, best_score)
                if (b<=a):
                    break
            
            return best_score

    #@staticmethod
    def get_ai_move(board):
        best_move = 0
        best_score = AI.INFINITE
        for move in board.legal_moves:
            board_clone = board.copy()
            board_clone.push(move)
            score = AI.alphabeta(board_clone, 3, -AI.INFINITE, AI.INFINITE, True)
            if (score < best_score):
                best_score = score
                best_move = move

        return best_move