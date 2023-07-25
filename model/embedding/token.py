import math

import torch
import torch.nn as nn


# class TokenEmbedding(nn.Embedding):
#     def __init__(self, vocab_size, embed_size=512):
#         super().__init__(vocab_size, embed_size, padding_idx=0)
#         nn.init.normal_(self.weight, mean=0, std=1/embed_size)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__()
        embed_weight = torch.randn(vocab_size, embed_size) * 0.01
        embed_weight[0, :] = 0.0
        self.embedding = nn.Embedding.from_pretrained(embed_weight, freeze=False, padding_idx=0)
        # self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # nn.init.normal_(self.embedding.weight, mean=0, std=1/embed_size)
        self.d_model = embed_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
