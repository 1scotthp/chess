from mingpt.model import GPT
import re
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader



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
    """ Encoder: Take a string of moves, return a list of integers. """
    return [stoi[move] for move in s.split()]

def decode(l):
    """ Decoder: Take a list of integers, return a string of moves. """
    return ' '.join([itos[i] for i in l])
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
print(len(data))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# one_encode = encode("1")
# dot_encode = encode(".")
money_encode = encode("$")
start_indices_train = []
for i in range(len(train_data) - block_size):
    segment = data[i:i + block_size]
    if [segment[0].item()] == money_encode:
        start_indices_train.append(i)

start_indices_val = []
for i in range(len(val_data) - block_size):
    segment = data[i:i + block_size]
    if [segment[0].item()] == money_encode:
        start_indices_val.append(i)

class TextDataset(Dataset):
    def __init__(self, data, start_indices, block_size):
        self.data = data
        self.start_indices = start_indices
        self.block_size = block_size

    def __len__(self):
        return len(self.start_indices)

    def get_vocab_size(self):
        return vocab_size

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        x = self.data[start_idx:start_idx + self.block_size].clone().detach().long()
        y = self.data[start_idx + 1:start_idx + self.block_size + 1].clone().detach().long()

        return x, y
    

import random
train_dataset = TextDataset(train_data, start_indices_train, block_size=block_size)
test_dataset = TextDataset(val_data, start_indices_val, block_size=block_size)

def estimate_loss(split, eval_iters):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    if split == 'test':
        dataset = test_dataset
    else: 
        dataset = train_dataset

    for k in range(eval_iters):
        # Randomly select an index
        idx = random.randrange(len(dataset))
        X, Y = dataset[idx]
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out
  
import chess
  # Function to find the first illegal move
def find_illegal_move(move_string, board):
    moves = move_string.split()
    legal_moves = 0
    for move in moves:
        try:
            # Try to make the move on the board
            board.push_san(move)
            legal_moves += 1
        except ValueError:
            # If ValueError is raised, the move is illegal
            return move, legal_moves
    return "All moves are legal", 0

from mingpt.trainer import Trainer
from transformers import AutoModel
if __name__ == '__main__':

    model_config = GPT.get_default_config()
    model_config.model_type = 'gopher-44m'
    model_config.vocab_size = vocab_size # openai's model vocabulary
    model_config.block_size = block_size  # openai's model block_size (i.e. input context length)
    model = GPT(model_config)

    # model = AutoModel.from_pretrained("name")
    


    train_config = Trainer.get_default_config()
    train_config.learning_rate = 1e-4 # many possible options, see the file
    train_config.max_iters = 500
    train_config.num_workers = 0
    train_config.batch_size = 8
    trainer = Trainer(train_config, model, train_dataset, test_dataset=test_dataset)
    max_runs = 5
    num_runs = 0
    while num_runs < max_runs:
        trainer.run()

        model.eval()
        with torch.no_grad():
            print(trainer.loss.item())
            dollar_token = stoi['$'] 
            weird_move = stoi['a3']
            inp = torch.tensor([[dollar_token, weird_move]], dtype=torch.long).to(trainer.device)
            # inp = encode("e4")
            with torch.no_grad():
                cat = model.generate(inp, block_size, do_sample=False)
                generated_text = decode(cat[0].tolist())[2:]
                print(generated_text)
                board = chess.Board()
                illegal_move, move_num = find_illegal_move(generated_text, board)
                print(f"The first illegal move is: {round(move_num/2) + 1}: {illegal_move}")

                cat = model.generate(inp, block_size, do_sample=False, temperature=1)
                generated_text = decode(cat[0].tolist())[2:]
                print(generated_text)
                board = chess.Board()
                illegal_move, move_num = find_illegal_move(generated_text, board)
                print(f"The first illegal move is: {round(move_num/2) + 1}: {illegal_move}")


        model.train()
        num_runs += 1 

    # torch.save(model.state_dict(), 'RUN.pth')
    model.to_pretrained("model")




# run a lot of examples from both train and test through the model and verify the output correctness
