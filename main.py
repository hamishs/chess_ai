import chess, random, ai

#return valid input move in UCI format
def get_valid_move(board):
	move_str = input("Your move:")
		
	try:
		move = chess.Move.from_uci(move_str)
	except Exception:
		print("Invalid syntax. Enter move in UCI format.")
		return get_valid_move

	if (not move in board.legal_moves):
		print("Move not legal.")
		return get_valid_move

	return move

# starting point
board = chess.Board()
print(board)

while True:
	#game over?
	if (board.is_game_over()):
		print(board.result())
		break

	#white move
	move = get_valid_move(board)
	board.push(move)
	print('White move made:')
	print(board)

	#game over?
	if (board.is_game_over()):
		print(board.result())
		break

	#black move
	print('Thinking...')
	ai_move = ai.AI.get_ai_move(board)
	board.push(ai_move)
	print('Black move made:')
	print(board)
