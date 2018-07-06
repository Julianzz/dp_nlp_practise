import torch 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import time
import random

import model 
import dictionary
import data


def train(dict1, dict2, input_tensor, target_tensor, encoder, decoder, encoder_optimizer,  \
    decoder_optimizer, criterion, max_length=data.MAX_LENGTH):
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    for si in range(input_length):
        try:
            encoder_output, encoder_hidden = encoder(input_tensor[si],encoder_hidden)
            encoder_outputs[si] = encoder_output[0,0]
        except:
            print("-----{}  {}".format(input_tensor[si],encoder.hidden_size))
            raise

    decoder_input = torch.tensor([[dict2.sos_index]])
    decoder_hidden = encoder_hidden

    loss = 0
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, \
            decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output,target_tensor[di])
        decoder_input = target_tensor[di]

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def train_iter(dict1, dict2, pairs, encoder, decoder, n_iters, learning_rate=0.01):
    start = time.time()
    
    encoder_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(),lr=learning_rate)
    training_pairs = [random.choice(pairs) for i in range(n_iters)]

    criterion = nn.NLLLoss()
    for iter in range(1, n_iters+1):
        training_pair = training_pairs[iter-1]
        input_tensor = data.tensor_from_sentence(dict1,training_pair[0])
        target_tensor = data.tensor_from_sentence(dict2,training_pair[1])

        loss = train(dict1, dict2, input_tensor, target_tensor, encoder,  \
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        if iter %10 == 0:
            print("%d %.4f"%(iter, loss))

    torch.save(encoder,"model/encoder.pth")
    torch.save(decoder, "model/decoder.pth")

import os

if __name__ == "__main__":
    
    dict1, dict2, pairs = data.load_data("eng", "fra")

    input_size = len(dict1.word_indexes)
    output_size = len(dict2.word_indexes)
    hidden_size = 8

    encoder_model_file = "model/encoder.pth"
    if os.path.exists(encoder_model_file):
        encoder = torch.load(encoder_model_file)
    else:
        encoder = model.RNNEncoder(input_size, hidden_size)

    decode_model_file = "model/decoder.pth"
    if os.path.exists(decode_model_file):
        decoder = torch.load(decode_model_file)
    else:
        decoder = model.AttnDecoder(hidden_size, output_size, data.MAX_LENGTH)

    train_iter(dict1, dict2, pairs, encoder, decoder, 2000)    
