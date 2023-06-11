import torch
import torch.nn as nn
import torch.optim as optim
from models.gru_model import GRU_Model
from main import train, test


# configure device
device = "cuda" if torch.cuda.is_available() else "cpu"

# rnn hyperparameter
epoch = 5
lr = 0.0001
input_size = 20076
hidden_size = 2000
hidden_size1 = 500
embedding_dim = 500
num_classes = 2
num_layers = 2


model = GRU_Model(input_size, embedding_dim, hidden_size, num_layers, hidden_size1, num_classes).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train(epoch, model, optimizer, criterion)
test(model, criterion)
