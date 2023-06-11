import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class GRU_Model(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, hidden_size1, num_layers, num_classes):
        super(GRU_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.RNN(embedding_dim, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, hidden_size1)
        self.fc1 = nn.Linear(hidden_size1, num_classes)

    def forward(self, x):
        out = self.embed(x)
        out, _ = self.gru(out, self.init_hidden(x.size(0)))
        out = self.relu(self.fc(out[:, 1]))
        out = self.fc1(out)
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
