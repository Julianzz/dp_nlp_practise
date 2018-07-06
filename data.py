import re
import torch
import dictionary
import unicodedata as unicodedata

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def index_from_sentence(dictionary,sentence):
    return [dictionary.word_to_index(w) for w in sentence.split(" ")]

def tensor_from_sentence(dictionary, sentence):
    indexes = index_from_sentence(dictionary, sentence)
    indexes.append(dictionary.eos_index)
    return torch.tensor(indexes,dtype=torch.long).view(-1, 1)

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i a ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def load_data(lang1, lang2, reverse=False,filter=True):
    # Read the file and split into lines
    with open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if filter:
        pairs = filter_pairs(pairs)

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_dict = dictionary.Dictionary(lang2)
        output_dict = dictionary.Dictionary(lang1)
        for line1, line2 in pairs:
            input_dict.load_sentence(line2)
            output_dict.load_sentence(line1)
    else:
        input_dict = dictionary.Dictionary(lang1)
        output_dict = dictionary.Dictionary(lang2)
        for line1, line2 in pairs:
            input_dict.load_sentence(line1)
            output_dict.load_sentence(line2)

    return input_dict, output_dict, pairs


if __name__ == "__main__":
    d = dictionary.Dictionary()
    line = "liu zhen zhong hi ni"
    d.load_sentence(line)

    t = tensor_from_sentence(d, "hi ni zhehong hao de")
    print(t)
    print(t.size())

    dict1, dict2, pairs = load_data("eng", "fra")
    print(len(dict1.word_indexes), len(dict2.word_indexes), len(pairs))
    for p1, p2 in pairs:
        print(p1, p2)
        break

