import re
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import chess

# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
block_size = 64 # what is the maximum context length for predictions?

with open('combined_games.txt', 'r', encoding='utf-8') as f:
    raw = f.read()


text = re.sub(r'\n', '$', raw)
text = re.sub(r'\$(?!1\.)', ' ', text)
text = text.replace('$$', ' ')
text = re.sub(r'\d+\.', ' ', text)
# print(text[:1000])
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
# Split the text into moves
moves = text.split()

# Create a set of unique moves
unique_moves = sorted(list(set(moves)))

# Determine the size of the vocabulary
vocab_size = len(unique_moves)
print(vocab_size)
# print(text[:1000])
stoi = {move: i for i, move in enumerate(unique_moves)}
itos = {i: move for i, move in enumerate(unique_moves)}
def encode(s):
    """ 
    Encoder: Take a string of moves, return a list of integers. 
    Ignores any tokens that cannot be encoded.
    """
    return [stoi[move] for move in s.split() if move in stoi]

def decode(l):
    """ Decoder: Take a list of integers, return a string of moves. """
    return ' '.join([itos[i] for i in l])


## input is 1 game of moves. output should be MxV matrix 

def generate_legal_moves(move_string, p = False):
    moves = move_string.split()
    board = chess.Board()
    legal_move_mat = []

    # print("moves", moves, p)
    for move in moves:
        try:
            # Try to make the move on the board
            board.push_san(move)
            legal_moves = board.legal_moves
            moves_str = ' '.join(str(board.san(move)) for move in legal_moves)
            # if p:
            #     print(p, "legal", moves_str)
            legal_move_mat.append(encode(moves_str))
        except ValueError:
            # If ValueError is raised, the move is illegal
            print(p, ValueError, move)
            
            break

    

    return legal_move_mat
    
    
    