
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
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
    def __init__(self, hidden_size, output_size):
        super(RNNDecoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output,hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoder(nn.Module):
    
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoder,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size,self.output_size)

    def forward(self, input, hidden, encoder_out, training=True):
        
        embeded = self.embedding(input).view(1, 1, -1)
        if training:
            embeded = self.dropout(embeded)
        # embeded -> (1, 1, embeded)

        cat = torch.cat((embeded[0], hidden[0]), 1)
        #print("cat:", cat.size())
        atten = self.attn(cat)
        #print("atten:", atten.size(), atten)
        atten_weight = F.softmax(atten, dim=1)
        atten_weight_un = atten_weight.unsqueeze(0)
        #print("atten_weight:", atten_weight.size(), atten_weight_un.size())

        encoder_out_un = encoder_out.unsqueeze(0)
    
        atten_applied = torch.bmm(atten_weight_un, encoder_out_un)
        #print("encoder_out:", encoder_out_un.size(), atten_weight_un.size(), atten_applied.size())

        output = torch.cat((embeded[0], atten_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, atten_weight

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