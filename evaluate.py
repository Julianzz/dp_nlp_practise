import torch

import data
import model
import dictionary

def evaluate(dict1, dict2, encoder, decoder, input_tensor, max_length=data.MAX_LENGTH):
    
    with torch.no_grad():
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_out, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] = encoder_out[0,0]

        decoder_input = torch.tensor([[dict2.sos_index]])

        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)
            #decoded_words.append(topi.item())

            if topi.item() == dict1.eos_index:
                decoded_words.append(dict1.eos_index)
                break
            
            decoded_words.append(topi.item())
            decoder_input = topi.squeeze().detach()
        
        return decoded_words, decoder_attentions[:di+1]


if __name__ == "__main__":

    dict1, dict2, pairs = data.load_data("eng", "fra")

    input_size = len(dict1.word_indexes)
    output_size = len(dict2.word_indexes)
    hidden_size = 8

    encoder = torch.load("model/encoder.pth")
    decoder = torch.load("model/decoder.pth")

    print(encoder.hidden_size)

    for p1, p2 in pairs:
        print(p1, "=====", p2)
        input_tensor = data.tensor_from_sentence(dict1,p1)
        target_tensor = data.tensor_from_sentence(dict2,p2)

        words, attentions = evaluate(dict1, dict2, encoder, decoder, input_tensor)
        #print(words, attentions)
        result = [ dict2.index_to_word(w) for w in words]

        print("======target:", p1)
        print("======predict:", " ".join(result))
        print("")