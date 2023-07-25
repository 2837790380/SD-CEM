import math

import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .hier import Embedding_layer
import torch
import numpy as np


class ModelEmbedding(nn.Module):
    """
    Model Embedding which is consisted with under features
        1. TokenEmbedding : embedding with hierarchical information
        2. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features are output of ModelEmbedding
    """

    def __init__(self, vocab_size, embed_size, hidden_dim, path_map, len_level, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        # self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.token = Embedding_layer(vocab_size=vocab_size, embed_size=embed_size,
                                     path_map=path_map, len_level=len_level)
        self.position = PositionalEmbedding(d_model=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

        self.bilstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_dim, num_layers=4,
                              batch_first=True, dropout=dropout, bidirectional=True)
        self.bilstm.flatten_parameters()

        self.affine = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear = nn.Linear(embed_size, hidden_dim)

    def forward(self, sequence, len_list):
        x = self.token(sequence)

        x = nn.LayerNorm(x.size()[1:]).to(sequence.device)(x)

        x = nn.utils.rnn.pack_padded_sequence(x, len_list.cpu(), batch_first=True,
                                              enforce_sorted=False).to(sequence.device)

        output, ht = self.bilstm(x)
        output, out_len = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=sequence.size(1))

        output = nn.LayerNorm(output.size()[1:]).to(sequence.device)(output)
        output = self.affine(output)

        output = output + self.position(sequence)
        return self.dropout(output)

