import torch
import torch.nn as nn
import math


class Embedding_layer(nn.Module):
    def __init__(self, vocab_size, embed_size, path_map, len_level, r=64, pad_idx=0):
        super(Embedding_layer, self).__init__()
        embed_weight = torch.randn(vocab_size, embed_size, requires_grad=True) * 0.001
        embed_weight[pad_idx, :] = 0.0
        self.venue_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.venue_embedding.weight = nn.Parameter(embed_weight)
        self.venue_embedding.weight.requires_grad = True

        self.Wa = nn.Linear(2 * embed_size, r)
        self.ua = torch.autograd.Variable(torch.randn(1, 1, 1, r), requires_grad=True)
        self.path_map = path_map
        self.len_level = len_level+1
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xs):
        eis = self.venue_embedding(xs)
        eis_ = torch.repeat_interleave(eis.unsqueeze(2), 6, dim=2)
        ejs = self.venue_embedding(torch.LongTensor(self.path_map[xs.cpu()]).to(xs.device))

        # ls = torch.Tensor(self.len_level[xs.cpu() - 4]).to(xs.device)
        # ls = torch.repeat_interleave(ls.unsqueeze(2), 6, dim=2)

        xs_mask = ejs == 0
        alpha_mask = ejs.sum(dim=-1) == 0
        # print(eis_.shape, xs_mask.shape)
        eis_ = eis_.masked_fill(xs_mask, 0)
        cat = torch.cat((eis_, ejs), dim=-1)
        score = torch.mul(self.ua.to(xs.device), self.Wa(cat)).sum(dim=-1)
        score = score.masked_fill(alpha_mask, -1e10)

        alpha = self.softmax(score)

        # gis = (torch.mul(alpha.unsqueeze(-1), ejs)).sum(-2) + eis  # sum(a * ejs) + ei, ejs 不包含ei
        # gis = (torch.mul(alpha.unsqueeze(-1), ejs) + eis_).sum(-2)  # + eis
        gis = torch.mul(alpha.unsqueeze(-1), ejs).sum(-2)  # sum(a * ejs), ejs 包含ei
        return gis

# class Embedding_layer(nn.Module):
#     def __init__(self, vocab_size, embed_size, path_map, len_level, pad_idx=0):
#         super(Embedding_layer, self).__init__()
#         embed_weight = torch.randn(vocab_size, embed_size, requires_grad=True) * 0.001
#         embed_weight[pad_idx, :] = 0.0
#         self.venue_embedding = nn.Embedding.from_pretrained(embed_weight, freeze=False, padding_idx=0)
#
#         self.embed_size = embed_size
#         self.path_map = path_map
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, xs):
#         eis = self.venue_embedding(xs)
#         eis_ = torch.repeat_interleave(eis.unsqueeze(2), 6, dim=2)
#         ejs = self.venue_embedding(torch.LongTensor(self.path_map[xs.cpu()]).to(xs.device))
#
#         xs_mask = ejs == 0
#         alpha_mask = ejs.sum(dim=-1) == 0
#         eis_ = eis_.masked_fill(xs_mask, 0)
#
#         score = (ejs * eis.unsqueeze(-2)) / math.sqrt(self.embed_size)
#         score = torch.sum(score, dim=-1)
#         score = score.masked_fill(alpha_mask, -1e10)
#         alpha = torch.softmax(score, dim=-1)
#
#         # gis = (torch.mul(alpha.unsqueeze(-1), ejs)).sum(-2) + eis  # sum(a * ejs) + ei, ejs 不包含ei
#         # gis = (torch.mul(alpha.unsqueeze(-1), ejs) + eis_).sum(-2)  # sum(a * ejs + ei)
#         gis = torch.mul(alpha.unsqueeze(-1), ejs).sum(-2)     # sum(a * ejs), ejs 包含ei
#
#         return gis