
import torch
import torch.nn as nn 

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
        return torch.zeros(1, 2, self.hidden_size)

class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, ouput_size):
        super(RNNDecoder,self).__init__()
        


if __name__ == "__main__":
    input_size = 20
    hidden_size = 8
    module = RNNEncoder(input_size, hidden_size)
    hidden = module.initHidden()
    input = torch.zeros(input_size)
    module(input,hidden)