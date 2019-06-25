#!/usr/bin/env python3
from flask import request, Flask
from flask import jsonify
from api import run_qasm
from terra import terra_move
from grover import grover_move
from vqc import vqc_move
import json

app = Flask(__name__)

@app.route('/')
def welcome():
    return "Hi Qiskiter!"

@app.route('/api/run/qasm', methods=['POST'])
def qasm():
    qasm = request.form.get('qasm')
    print("--------------")
    print (qasm)
    print(request.get_data())
    print (request.form)
    backend = 'qasm_simulator'
    output = run_qasm(qasm, backend)
    ret = {"result": output}
    return jsonify(ret)

@app.route('/tictactoe/player/terra', methods=['POST'])
def terra():
    board = request.form.get('board')
    board = board.split(',')

    (move, t_counts) = terra_move(board)
    ret = {"move": move, "t_counts": t_counts}

    print('sending : ', ret)
    return jsonify(ret)

@app.route('/tictactoe/player/grover', methods=['POST'])
def grover():

    print('GROVER');
    board = request.form.get('board')
    board = board.split(',')

    move = grover_move(board)
    ret = {"move": move}

    print('sending : ', ret)
    return jsonify(ret)

@app.route('/tictactoe/player/vqc', methods=['POST'])
def vqc():

    print('VQC');
    board = request.form.get('board')
    board = board.split(',')

    move = vqc_move(board)
    ret = {"move": move}

    print('sending : ', ret)
    return jsonify(ret)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
