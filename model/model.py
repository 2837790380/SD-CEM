import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import ModelEmbedding


class Model(nn.Module):

    def __init__(self, mob, path_map, len_level, vocab_size, embedding_dim=30, hidden_dim=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden_dim: model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden_dim
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.mob = mob

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden_dim * 4

        # embedding, sum of positional, segment, token embeddings
        self.embedding = ModelEmbedding(vocab_size=vocab_size, embed_size=embedding_dim,
                                       hidden_dim=hidden_dim, path_map=path_map, len_level=len_level, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_block = TransformerBlock(hidden_dim, attn_heads, hidden_dim*4, dropout)

    def forward(self, x, len_list):
        # attention masking for padded token
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, len_list)

        # running over multiple transformer blocks
        for _ in range(self.n_layers):
            x = self.transformer_block(x, mask)

        return x

