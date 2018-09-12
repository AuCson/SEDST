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


class DynamicEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout, bidir=True, extra_size=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size + extra_size, hidden_size, n_layers, bidirectional=bidir)
        self.gru = orth_gru(self.gru)
        self.bidir = bidir

    def forward(self, input_seqs, input_lens, hidden=None, seperate=False, return_emb=False, extra_input=None):
        """
        forward procedure. No need for inputs to be sorted
        :param extra_input: [T, B, E]
        :param return_emb:
        :param seperate:
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded_origin = self.embedding(input_seqs)
        embedded = embedded_origin.transpose(0, 1)  # [B,T,E]
        if extra_input is not None:
            extra_input = extra_input.transpose(0,1) # [B,T,E]
        if hidden is not None:
            hidden = hidden.transpose(0,1) # [B,L,H]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        if extra_input is not None:
            extra_input = extra_input[sort_idx].transpose(0,1) # [T,B,E]
            inp = torch.cat([embedded, extra_input], -1)
        else:
            inp = embedded
        if hidden is not None:
            hidden = hidden[sort_idx].transpose(0,1) # [L,B,H]
        packed = torch.nn.utils.rnn.pack_padded_sequence(inp, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        if self.bidir and not seperate:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        if not return_emb:
            return outputs, hidden
        else:
            return outputs, hidden, embedded_origin

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


class Attn(nn.Module):
    """
    compute attention vector (1 layer)
    """

    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.zeros(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        # self.ln1 = LayerNormalization(hidden_size)

    def forward(self, hidden, encoder_outputs, normalize=True):
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(hidden, encoder_outputs)
        normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
        context = torch.bmm(normalized_energy, encoder_outputs).transpose(0, 1)  # [1,B,H]
        # context = self.ln1(context)
        return context

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        assert hidden.size(0) == 1
        H = hidden.expand(max_len, -1, -1).transpose(0, 1)
        energy = F.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy