
import torch
import torch.nn as nn
import torch.functional as F

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1,1,-1)
        output = embed
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, ouput_size):
        super(RNNDecoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(ouput_size, hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size, ouput_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        ouput = self.embedding(input).view(1, 1, -1)
        ouput = F.relu(ouput)
        ouput, hidden = self.gru(ouput,hidden)
        ouput = self.softmax(self.out(ouput[0]))
        return ouput, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

if __name__ == "__main__":
    input_size = 20
    hidden_size = 8
    module = RNNEncoder(input_size, hidden_size)
    hidden = module.initHidden()
    input = torch.zeros(1, 1).long()
    print(input)
    out, hidden = module(input,hidden)
    print(out.size())
    print(hidden.size())