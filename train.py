import torch 
import torch.functional as F
import torch.nn as nn
import torch.optim as optim

import time
import random

import model 
import dictionary
import data

MAX_LENGTH = 1200

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,  \
    decoder_optimizer, criterion, max_length=MAX_LENGTH):
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    for si in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[si],encoder_hidden)
        encoder_outputs[si] = encoder_output[0,0]

    decoder_input = torch.tensor([[dictionary.SOS_token]])
    decoder_hidden = encoder_hidden

    loss = 0
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, \
            decoder_hidden,encoder_outputs)
        loss += criterion(decoder_output,target_tensor[di])
        decoder_input = target_tensor[di]

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate(encoder, decoder,sentence, max_length=MAX_LENGTH):
    
    with torch.no_grad():
        input_tensor = []
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_out, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] = encoder_out[0,0]

        decoder_input = torch.tensor([[dictionary.SOS_token]])

        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(topi.item())


            """if topi.item() == dictionary.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            """

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

        loss = train(input_tensor, target_tensor, encoder,  \
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        if iter %10 == 0:
            print("%d %.4f\n", iter, loss)


if __name__ == "__main__":
    dict1, dict2, pairs = data.load_data("eng", "fra")

