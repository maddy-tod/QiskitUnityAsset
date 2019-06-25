

// TODO need to make it set up the VQC at the start
def vqc_move(board):
    board = [x if x else 0 for x in board]

    to_predict = VQCQPlayer.singleDataItem(data_path, data_file, board, n=feature_dim)

    move = self.algo_obj.predict(to_predict)[0]

    # if the move selected already contains a play, choose randomly
    if board[move]:
        spaces = [index for index, x in enumerate(board) if x == 0]
        move = random.choice(spaces)

    return move
