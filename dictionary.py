class Dictionary(object):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.eos = "<EOS>"
        self.sos = "<SOS>"
        self.unk = "<UNK>"

        self.words = [self.eos, self.sos, self.unk]
        self.word_indexes = {}
        for i, word in enumerate(self.words):
            self.word_indexes[word] = i
    
    @property
    def eos_index(self):
        return self.word_indexes[self.eos]

    @property
    def sos_index(self):
        return self.word_indexes[self.sos]
    
    @property
    def unk_index(self):
        return self.word_indexes[self.unk]
    
    def load_sentence(self, line):
        words = line.split(" ")
        for item in words:
            item = item.strip()
            if not item:
                return 
            self.add_word(item)

    def add_word(self,word): 
        if not word in self.word_indexes:
            index = len(self.words) 
            self.words.append(word)
            self.word_indexes[word] = index

    def word_to_index(self, word):
        if word not in self.word_indexes:
            return self.word_indexes[self.unk]
        return self.word_indexes[word]

    def index_to_word(self, index):
        return self.words[index]