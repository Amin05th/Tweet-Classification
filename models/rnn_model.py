import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class RNN_Model(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, hidden_size1, hidden_size2, num_layers, num_classes):
        super(RNN_Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(hidden_size, hidden_size1)
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.embed(x)
        out, _ = self.rnn(out, self.init_hidden(x.size(0)))
        out = self.relu(self.fc(out[:, 1]))
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
