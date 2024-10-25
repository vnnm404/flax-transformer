class MultiplicationTokenizer:
    def __init__(self):
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update(
            {
                "x": 10,
                "=": 11,
                "(": 12,
                ")": 13,
                "+": 14,
                "b": 15,
                "e": 16,
                "p": 17,
                "u": 18,
                "t": 19,
                "ts": 20,
            }
        )
        self.bos_token = "b"
        self.eos_token = "e"
        self.pad_token = "p"
        self.unk_token = "u"
        self.think_token = "t"
        self.think_symbol_token = "ts"

        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.think_token_id = self.vocab[self.think_token]
        self.think_symbol_token_id = self.vocab[self.think_symbol_token]

        self.id2token = {v: k for k, v in self.vocab.items()}

        self.vocab_size = self.__len__()

    def encode(self, text, max_length=None, pad=True):
        tokens = list(text)
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]

        if max_length is not None:
            if pad and len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids

    def decode(self, ids):
        # tokens = [self.id2token[id] for id in ids if id != self.pad_token_id]
        tokens = [self.id2token[id] for id in ids]
        return "".join(tokens)

    def add_token(self, token):
        if token not in self.vocab:
            next_id = len(self.vocab)
            self.vocab[token] = next_id
            self.id2token[next_id] = token

    def __len__(self):
        return len(self.vocab)
