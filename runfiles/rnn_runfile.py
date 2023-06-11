import torch
import torch.nn as nn
import torch.optim as optim
from models.rnn_model import RNN_Model
from main import train, test


# configure device
device = "cuda" if torch.cuda.is_available() else "cpu"

# rnn hyperparameter
epoch = 5
lr = 0.0001
input_size = 20076
hidden_size = 3500
hidden_size1 = 2500
hidden_size2 = 1500
embedding_dim = 1500
num_classes = 2
num_layers = 2


model = RNN_Model(input_size, embedding_dim, hidden_size, hidden_size1, hidden_size2, num_layers, num_classes).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train(epoch, model, optimizer, criterion)
test(model, criterion)

