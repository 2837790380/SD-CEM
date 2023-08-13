import torch
import torch.nn as nn
import torch.nn.functional as f

from .model import Model
import numpy as np


class SDCEM(nn.Module):
    """
    SD-CEM
    Transition Prediction Model + Masked Category Model
    """

    def __init__(self, model: Model, vocab_size):
        """
        :param model: model which should be trained
        :param vocab_size: total vocab size
        """

        super().__init__()
        self.model = model

        self.category_ids = torch.LongTensor(np.arange(0, vocab_size).reshape(-1, 1))
        self.mask_lm = MaskedCategoryModel(self.model.hidden, vocab_size)
        self.next_tm = NextTransitionPrediction(self.model, self.category_ids, self.model.mob)

        self.dropout = nn.Dropout(0.5)
        self.device = None

    def forward(self, data_seqs, s_ids, d_ids, len_list):
        x = self.model(data_seqs, len_list)
        self.device = data_seqs.device

        s_embeddings = self.model.embedding.token(s_ids)
        all_embeddings = self.model.embedding.token(self.category_ids.to(self.device))

        return self.mask_lm(x), self.next_tm(self.device, s_ids, s_embeddings, all_embeddings)


def pairwise_inner_product(embed1, embed2):
    num_embed = embed1.size()[0]
    mat_expand = embed1.unsqueeze(dim=0).repeat(num_embed, 1, 1)  # num_embed x num_embed x embed_dim
    mat_expand = torch.transpose(mat_expand, 0, 1)

    inner_prod = torch.mul(mat_expand, embed2)  # num_embed x num_embed x embed_dim
    inner_prod = inner_prod.sum(-1)

    return inner_prod


class NextTransitionPrediction(nn.Module):
    def __init__(self, model: Model, category_ids, mob):
        super(NextTransitionPrediction, self).__init__()
        self.model = model
        self.softmax = nn.LogSoftmax(-1)
        self.category_ids = category_ids
        self.mob = mob
        self.mob_weight = nn.Embedding.from_pretrained(self.mob, freeze=True)

    def forward(self, device, s_ids, s_embeddings, all_embeddings):
        s2d_mob = self.mob_weight(s_ids)
        s2d = torch.matmul(s_embeddings, all_embeddings.squeeze(-2).transpose(0, 1))
        p_sd = s2d_mob * self.softmax(s2d)

        return p_sd


class MaskedCategoryModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
