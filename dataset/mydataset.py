from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data, next_data, labelofseq, vocab_size, seq_len, encoding="utf-8"):
        self.data = data
        self.next_data = next_data
        self.labelofseq = labelofseq
        self.seq_len = seq_len
        self.encoding = encoding
        self.vocab_size = vocab_size
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        t = self.data[item]
        t_next = self.next_data[item]
        t_random, t_label = self.random_word(t)
        length = len(t_random)

        bert_input = list(t_random[:self.seq_len])
        next_input = list(t_next[:self.seq_len])
        t = list(t[:self.seq_len])

        bert_label = list(t_label[:self.seq_len])
        lenofseq = min(len(bert_input), self.seq_len)
        
        padding = [0 for _ in range(self.seq_len - length)]
        bert_input.extend(padding), next_input.extend(padding), bert_label.extend(padding), t.extend(padding)

        output = {"input": bert_input,
                  "source_input": t,
                  "next_input": next_input,
                  "label": bert_label,
                  "length": lenofseq}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token, [mask]:3
                if prob < 0.8:
                    tokens[i] = 3

                # 10% randomly change token to random token, 0-3 are special tokens
                elif prob < 0.9:
                    tokens[i] = random.randrange(4, self.vocab_size)

                # 10% randomly change token to current token
                else:
                    tokens[i] = tokens[i]
                    # continue

                output_label.append(token)

            else:
                tokens[i] = token
                output_label.append(0)

        return tokens, output_label