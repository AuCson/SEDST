import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from config import global_config as cfg


def cuda_(var, aux=None):
    if not aux:
        return var.cuda() if cfg.cuda else var
    elif aux != 'cpu' and aux >= 0 and cfg.cuda:
        return var.cuda(aux)
    else:
        return var.cpu()


def orth_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + gru.hidden_size], gain=1)
    return gru

class LayerNormalization(nn.Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out


class DynamicEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(self.hidden_size, 1)

    def forward(self, hidden, encoder_outputs, normalize=True):
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(hidden, encoder_outputs)
        normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        return context.transpose(0, 1)  # [1,B,H]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        energy = self.attn(torch.cat([H, encoder_outputs], 2))  # [B,T,2H]->[B,T,H]
        # fix attention here
        energy = self.v(F.tanh(energy)).transpose(1,2) # [B,1,T]
        return energy