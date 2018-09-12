import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
from config import global_config as cfg
import copy, random, time, logging


def cuda_(var):
    return var.cuda() if cfg.cuda else var


def toss_(p):
    return random.randint(0, 99) <= p


def nan(v):
    return np.isnan(np.sum(v.data.cpu().numpy()))


def get_sparse_input(x_input):
    """
    get a sparse matrix of x_input: [T,B,V] where x_sparse[i][j][k]=1, and others = 1e-8
    :param x_input: *Tensor* of [T,B]
    :return: *Tensor* in shape [B,T,V]
    """
    # indexes that will make no effect in copying
    sw = time.time()
    print('sparse input start: %s' % sw)
    ignore_index = [0]
    result = torch.normal(mean=0, std=torch.zeros(x_input.size(0), x_input.size(1), cfg.vocab_size))
    for t in range(x_input.size(0)):
        for b in range(x_input.size(1)):
            if x_input[t][b] not in ignore_index:
                result[t][b][x_input[t][b]] = 1.0
    print('sparse input end %s' % time.time())
    return result.transpose(0, 1)


def get_sparse_input_efficient(x_input_np):
    ignore_index = [0]
    result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size), dtype=np.float32)
    result.fill(1e-10)
    for t in range(x_input_np.shape[0]):
        for b in range(x_input_np.shape[1]):
            if x_input_np[t][b] not in ignore_index:
                result[t][b][x_input_np[t][b]] = 1.0
    result_np = result.transpose((1, 0, 2))
    result = torch.from_numpy(result_np).float()
    return result


def shift(pz_proba):
    first_input = np.zeros((pz_proba.size(1), pz_proba.size(2)))
    first_input.fill(1e-10)
    first_input = cuda_(Variable(torch.from_numpy(first_input)).float())
    pz_proba = list(pz_proba)[:-1]
    pz_proba.insert(0, first_input)
    pz_proba = torch.stack(pz_proba, 0)
    return pz_proba.contiguous()


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, hidden=None):
        embedded = self.embedding(input_seqs)
        outputs, hidden = self.gru(embedded, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


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

        self.initial_hidden = nn.Parameter(torch.zeros(1, 1, hidden_size))
        torch.nn.init.orthogonal(self.initial_hidden)
        self.initial_hidden.requires_grad = False

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        if hidden is None:
            hidden = self.initial_hidden.repeat(self.n_layers * 2, batch_size, 1)
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
        self.v = nn.Parameter(torch.zeros(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

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
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy


class MultiTurnInferenceDecoder_Z(nn.Module):
    """
    Inference network: copying version of Q_phi(z_t|s_t,m_t) <- Q_phi(z_ti|s_t,m_t,z_t[1..i-1])
    """

    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate):
        super().__init__()
        self.gru = nn.GRU(embed_size, hidden_size, dropout=dropout_rate)
        self.w1 = nn.Linear(hidden_size, vocab_size)
        self.mu = nn.Linear(vocab_size, embed_size)
        self.log_sigma = nn.Linear(vocab_size, embed_size)
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, u_input, u_enc_out, pv_pz_proba, pv_z_dec_out, m_input, m_enc_out, embed_z, last_hidden,
                rand_eps, u_input_np, m_input_np):
        """
        Similar to base class method
        :param m_input:
        :param u_input:
        :param u_enc_out:
        :param m_enc_out:
        :param embed_z:
        :param last_hidden:
        :param rand_eps:
        :return:
        """
        sparse_u_input = Variable(get_sparse_input_efficient(u_input_np), requires_grad=False)  # [B,T,V]
        sparse_m_input = Variable(get_sparse_input_efficient(m_input_np), requires_grad=False)  # [B,T,V]

        # if cfg.cuda: sparse_m_input = sparse_m_input.cuda()
        # if cfg.cuda: sparse_u_input = sparse_u_input.cuda()

        embed_z = F.dropout(embed_z, self.dropout_rate)
        gru_out, last_hidden = self.gru(embed_z, last_hidden)
        gen_score = self.w1(gru_out).squeeze(0) # [B,V]
        u_copy_score = F.tanh(self.proj_copy1(u_enc_out.transpose(0, 1)))  # [B,T,H]
        m_copy_score = F.tanh(self.proj_copy2(m_enc_out.transpose(0, 1)))
        if not cfg.force_stable:
            # unstable version of copynet for small dataset
            u_copy_score = torch.exp(torch.matmul(u_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2))  # [B,T]
            m_copy_score = torch.exp(torch.matmul(m_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2))  # [B,T]
            u_copy_score, m_copy_score = u_copy_score.cpu(), m_copy_score.cpu()
            u_copy_score = torch.log(torch.bmm(u_copy_score.unsqueeze(1), sparse_u_input)).squeeze(1)  # [B,V]
            m_copy_score = torch.log(torch.bmm(m_copy_score.unsqueeze(1), sparse_m_input)).squeeze(1)  # [B,V]
        else:
            # stable version of copynet
            u_copy_score = torch.matmul(u_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
            m_copy_score = torch.matmul(m_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
            u_copy_score, m_copy_score = u_copy_score.cpu(), m_copy_score.cpu()
            u_copy_score_max, m_copy_score_max = torch.max(u_copy_score, dim=1, keepdim=True)[0], \
                                                 torch.max(m_copy_score, dim=1, keepdim=True)[0]
            u_copy_score = torch.exp(u_copy_score - u_copy_score_max)  # [B,T]
            m_copy_score = torch.exp(m_copy_score - m_copy_score_max)  # [B,T]
            # u_copy_score, m_copy_score = u_copy_score.cpu(), m_copy_score.cpu()
            u_copy_score = torch.log(torch.bmm(u_copy_score.unsqueeze(1), sparse_u_input)).squeeze(
                1) + u_copy_score_max  # [B,V]
            m_copy_score = torch.log(torch.bmm(m_copy_score.unsqueeze(1), sparse_m_input)).squeeze(
                1) + m_copy_score_max  # [B,V]
        u_copy_score, m_copy_score = cuda_(u_copy_score), cuda_(m_copy_score)
        if pv_pz_proba is not None:
            pv_pz_proba = shift(pv_pz_proba)
            pv_z_copy_score = F.tanh(self.proj_copy3(pv_z_dec_out.transpose(0, 1)))  # [B,T,H]
            if cfg.force_stable:
                pv_z_copy_score = torch.exp(
                    torch.matmul(pv_z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2))  # [B,T]
                pv_z_copy_score = torch.log(
                    torch.bmm(pv_z_copy_score.unsqueeze(1), pv_pz_proba.transpose(0, 1))).squeeze(
                    1)  # [B,V]
            else:
                pv_z_copy_score = torch.matmul(pv_z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
                pv_z_copy_score_max = torch.max(pv_z_copy_score, dim=1, keepdim=True)[0]
                pv_z_copy_score = torch.exp(pv_z_copy_score - pv_z_copy_score_max)
                pv_z_copy_score = torch.log(
                    torch.bmm(pv_z_copy_score.unsqueeze(1), pv_pz_proba.transpose(0, 1))).squeeze(
                    1) + pv_z_copy_score_max  # [B,V]
            scores = F.softmax(torch.cat([gen_score, u_copy_score, m_copy_score, pv_z_copy_score], dim=1), dim=1)
            gen_score, u_copy_score, m_copy_score, pv_z_copy_score = tuple(
                torch.split(scores, gen_score.size(1), dim=1))
            proba = gen_score + u_copy_score + m_copy_score + pv_z_copy_score
        else:
            scores = F.softmax(torch.cat([gen_score, u_copy_score, m_copy_score], dim=1), dim=1)
            gen_score, u_copy_score, m_copy_score = tuple(
                torch.split(scores, gen_score.size(1), dim=1))
            proba = gen_score + u_copy_score + m_copy_score
        appr_emb = self.mu(proba)
        log_sigma_ae = self.log_sigma(proba)
        sigma_ae = torch.exp(log_sigma_ae)
        sampled_ae = appr_emb + torch.mul(sigma_ae, rand_eps)
        return sampled_ae, gru_out, last_hidden, proba, appr_emb, log_sigma_ae


class MultiTurnPriorDecoder_Z(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate):
        super().__init__()
        self.gru = nn.GRU(embed_size, hidden_size, dropout=dropout_rate)
        self.w1 = nn.Linear(hidden_size, vocab_size)
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(vocab_size, embed_size)
        self.log_sigma = nn.Linear(vocab_size, embed_size)
        self.dropout_rate = dropout_rate

    def forward(self, u_input, u_enc_out, pv_pz_proba, pv_z_dec_out, embed_z, last_hidden, rand_eps, u_input_np,
                m_input_np):
        sparse_u_input = Variable(get_sparse_input_efficient(u_input_np), requires_grad=False)
        embed_z = F.dropout(embed_z, self.dropout_rate)
        gru_out, last_hidden = self.gru(embed_z, last_hidden)
        gen_score = self.w1(gru_out).squeeze(0)
        u_copy_score = F.tanh(self.proj_copy1(u_enc_out.transpose(0, 1)))  # [B,T,H]
        if not cfg.force_stable:
            u_copy_score = torch.exp(torch.matmul(u_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2))  # [B,T]
            u_copy_score = u_copy_score.cpu()
            u_copy_score = torch.log(torch.bmm(u_copy_score.unsqueeze(1), sparse_u_input)).squeeze(1)  # [B,V]
        else:
            # stable version of copynet
            u_copy_score = torch.matmul(u_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
            u_copy_score = u_copy_score.cpu()
            u_copy_score_max = torch.max(u_copy_score, dim=1, keepdim=True)[0]
            u_copy_score = torch.exp(u_copy_score - u_copy_score_max)  # [B,T]
            u_copy_score = torch.log(torch.bmm(u_copy_score.unsqueeze(1), sparse_u_input)).squeeze(
                1) + u_copy_score_max  # [B,V]
        u_copy_score = cuda_(u_copy_score)
        if pv_pz_proba is not None:
            pv_pz_proba = shift(pv_pz_proba)
            pv_z_copy_score = F.tanh(self.proj_copy2(pv_z_dec_out.transpose(0, 1)))  # [B,T,H]
            if cfg.force_stable:
                pv_z_copy_score = torch.exp(
                    torch.matmul(pv_z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2))  # [B,T]
                pv_z_copy_score = torch.log(
                    torch.bmm(pv_z_copy_score.unsqueeze(1), pv_pz_proba.transpose(0, 1))).squeeze(
                    1)  # [B,V]
            else:
                pv_z_copy_score = torch.matmul(pv_z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
                pv_z_copy_score_max = torch.max(pv_z_copy_score, dim=1, keepdim=True)[0]
                pv_z_copy_score = torch.exp(pv_z_copy_score - pv_z_copy_score_max)
                pv_z_copy_score = torch.log(
                    torch.bmm(pv_z_copy_score.unsqueeze(1), pv_pz_proba.transpose(0, 1))).squeeze(
                    1) + pv_z_copy_score_max  # [B,V]
            scores = F.softmax(torch.cat([gen_score, u_copy_score, pv_z_copy_score], dim=1), dim=1)
            gen_score, u_copy_score, pv_z_copy_score = tuple(torch.split(scores, gen_score.size(1), dim=1))
            proba = gen_score + u_copy_score + pv_z_copy_score  # [B,V]
        else:
            scores = F.softmax(torch.cat([gen_score, u_copy_score], dim=1), dim=1)
            gen_score, u_copy_score = tuple(torch.split(scores, gen_score.size(1), dim=1))
            proba = gen_score + u_copy_score  # [B,V]
        appr_emb = self.mu(proba)
        log_sigma_ae = self.log_sigma(proba)
        sigma_ae = torch.exp(log_sigma_ae)
        sampled_ae = appr_emb + torch.mul(sigma_ae, rand_eps)
        return sampled_ae, gru_out, last_hidden, proba, appr_emb, log_sigma_ae


class ResponseDecoder(nn.Module):
    """
    Response decoder: P_theta(m_t|s_t, z_t) <- P_theta(m_ti|s_t, z_t, m_t[1..i-1])
    This is a deterministic decoder.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, dropout_rate):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.attn_z = Attn(hidden_size)
        self.attn_u = Attn(hidden_size)
        self.w4 = nn.Linear(hidden_size, hidden_size)
        self.gate_z = nn.Linear(hidden_size, hidden_size)
        self.w5 = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size + degree_size, hidden_size, dropout=dropout_rate)
        self.proj = nn.Linear(hidden_size * 3, vocab_size)
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate

    def forward(self, z_enc_out, pz_proba, u_enc_out, m_t_input, degree_input, last_hidden):
        """
        decode the response: P(m|u,z)
        :param degree_input: [B,D]
        :param pz_proba: [Tz,B,V], output of the prior decoder
        :param z_enc_out: [Tz,B,H]
        :param u_enc_out: [T,B,H]
        :param m_t_input: [1,B]
        :param last_hidden:
        :return: proba: [1,B,V]
        """
        m_embed = self.emb(m_t_input)
        pz_proba = shift(pz_proba)

        z_context = self.attn_z(last_hidden, z_enc_out)
        u_context = self.attn_u(last_hidden, u_enc_out)
        d_control = self.w4(z_context) + torch.mul(F.sigmoid(self.gate_z(z_context)), self.w5(u_context))
        gru_out, last_hidden = self.gru(torch.cat([d_control, m_embed, degree_input.unsqueeze(0)], dim=2), last_hidden)
        gen_score = self.proj(torch.cat([z_context, u_context, gru_out], 2)).squeeze(0)
        z_copy_score = F.tanh(self.proj_copy1(z_enc_out.transpose(0, 1)))  # [B,T,H]
        if not cfg.force_stable:
            z_copy_score = torch.exp(torch.matmul(z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2))  # [B,T]
            z_copy_score = torch.log(torch.bmm(z_copy_score.unsqueeze(1), pz_proba.transpose(0, 1))).squeeze(1)  # [B,V]
        else:
            z_copy_score = torch.matmul(z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
            z_copy_score_max = torch.max(z_copy_score, dim=1, keepdim=True)[0]
            z_copy_score = torch.exp(z_copy_score - z_copy_score_max)
            z_copy_score = torch.log(torch.bmm(z_copy_score.unsqueeze(1), pz_proba.transpose(0, 1)))
            z_copy_score = z_copy_score.squeeze(1) + z_copy_score_max
        scores = F.softmax(torch.cat([gen_score, z_copy_score], dim=1), dim=1)
        gen_score, z_copy_score = tuple(torch.split(scores, gen_score.size(1), dim=1))
        proba = gen_score + z_copy_score  # [B,V]
        return proba, last_hidden, gru_out


class MultinomialKLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p_proba, q_proba): # [B, T, V]
        mask = torch.zeros(p_proba.size(0), p_proba.size(1))
        for i in range(p_proba.size(0)):
            for j in range(q_proba.size(0)):
                topv, topi = torch.max(p_proba[i,j], -1)
                if topi.item() == 0:
                    mask[i,j] = 0
                else:
                    mask[i,j] = 1
        mask = cuda_(Variable(mask))
        loss = q_proba * (torch.log(q_proba) - torch.log(p_proba))
        masked_loss = torch.sum(mask.unsqueeze(-1) * loss)
        return masked_loss / (p_proba.size(1) * p_proba.size(0))


class SemiSupervisedSEDST(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, layer_num, dropout_rate, z_length, alpha,
                 max_ts, beam_search=False, teacher_force=100, **kwargs):
        super().__init__()
        self.u_encoder = DynamicEncoder(vocab_size, embed_size, hidden_size, layer_num, dropout_rate)
        self.m_encoder = DynamicEncoder(vocab_size, embed_size, hidden_size, layer_num, dropout_rate)
        self.qz_decoder = MultiTurnInferenceDecoder_Z(embed_size, hidden_size, vocab_size, dropout_rate)  # posterior
        self.pz_decoder = MultiTurnPriorDecoder_Z(embed_size, hidden_size, vocab_size, dropout_rate)  # prior
        self.m_decoder = ResponseDecoder(embed_size, hidden_size, vocab_size, degree_size, dropout_rate)
        self.embed_size = embed_size
        self.vocab = kwargs['vocab']

        self.pr_loss = nn.NLLLoss(ignore_index=0)
        self.q_loss = nn.NLLLoss(ignore_index=0)
        self.dec_loss = nn.NLLLoss(ignore_index=0)
        self.kl_loss = MultinomialKLDivergenceLoss()

        self.z_length = z_length
        self.alpha = alpha
        self.max_ts = max_ts
        self.beam_search = beam_search
        self.teacher_force = teacher_force

        if self.beam_search:
            self.beam_size = kwargs['beam_size']
            self.eos_token_idx = kwargs['eos_token_idx']

    def forward(self, u_input, u_input_np, m_input, m_input_np, z_input, u_len, m_len, turn_states, z_supervised,
                p_input, p_input_np, p_len,
                degree_input, mode):
        if mode == 'train' or mode == 'valid':
            if not z_supervised:
                z_input = None
            pz_proba, qz_proba, pm_dec_proba, pz_mu, pz_log_sigma, qz_mu, qz_log_sigma, turn_states = \
                self.forward_turn(u_input, u_len, m_input=m_input, m_len=m_len, z_input=z_input, is_train=True,
                                  turn_states=turn_states, degree_input=degree_input, u_input_np=u_input_np,
                                  m_input_np=m_input_np,p_input=p_input,p_input_np=p_input_np,p_len=p_len)
            if z_supervised:
                loss, pr_loss, m_loss, q_loss = self.supervised_loss(torch.log(pz_proba), torch.log(qz_proba),
                                                                     torch.log(pm_dec_proba), z_input, m_input)
                return loss, pr_loss, m_loss, q_loss, turn_states
            else:
                loss, m_loss, kl_div_loss = self.unsupervised_loss(qz_mu, qz_log_sigma, pz_mu, pz_log_sigma,
                                                                   torch.log(pm_dec_proba), m_input, pz_proba, qz_proba)
            return loss, m_loss, kl_div_loss, turn_states
        elif mode == 'test':
            m_output_index, pz_index, turn_states = self.forward_turn(u_input, u_len=u_len, is_train=False,
                                                                      turn_states=turn_states,
                                                                      degree_input=degree_input,
                                                                      u_input_np=u_input_np, m_input_np=m_input_np,
                                                                      p_input=p_input, p_input_np=p_input_np,
                                                                      p_len=p_len
                                                                      )
            return m_output_index, pz_index, turn_states


    def forward_turn(self, u_input, u_len, turn_states, is_train, degree_input, u_input_np, m_input_np=None,
                     m_input=None, m_len=None, z_input=None,
                     p_input=None, p_input_np=None, p_len=None,test_type='pr'):
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        :param u_input_np:
        :param m_input_np:
        :param u_len:
        :param turn_states:
        :param is_train:
        :param u_input: [T,B]
        :param m_input: [T,B]
        :param z_input: [T,B]
        :return:
        """
        pv_pz_proba = turn_states.get('pv_pz_proba', None)
        pv_z_outs = turn_states.get('pv_z_dec_outs', None)
        batch_size = u_input.size(1)
        u_enc_out, u_enc_hidden = self.u_encoder(u_input, u_len)
        last_hidden = u_enc_hidden[:-1]
        # initial approximate embedding: SOS token initialized with all zero
        # Pi(z|u)
        pz_ae = cuda_(Variable(torch.zeros(1, batch_size, self.embed_size)))
        pz_proba, pz_mu, pz_log_sigma = [], [], []
        pz_dec_outs = []
        z_length = z_input.size(0) if z_input is not None else self.z_length
        for t in range(z_length):
            if cfg.sampling:
                rand_eps = Variable(torch.normal(means=torch.zeros(1, batch_size, cfg.embedding_size), std=1))
            else:
                rand_eps = Variable(torch.zeros(1, batch_size, cfg.embedding_size))
            if cfg.cuda: rand_eps = rand_eps.cuda()
            pz_ae, last_hidden, pz_dec_out, proba, appr_emb, log_sigma_ae = \
                self.pz_decoder(u_input=u_input, u_enc_out=u_enc_out, pv_pz_proba=pv_pz_proba, pv_z_dec_out=pv_z_outs,
                                embed_z=pz_ae, last_hidden=last_hidden, rand_eps=rand_eps, u_input_np=u_input_np,
                                m_input_np=m_input_np)
            pz_proba.append(proba)
            pz_mu.append(appr_emb)
            pz_log_sigma.append(log_sigma_ae)
            pz_dec_outs.append(pz_dec_out)
        pz_dec_outs = torch.cat(pz_dec_outs, dim=0)  # [Tz,B,H]
        pz_proba, pz_mu, pz_log_sigma = torch.stack(pz_proba, dim=0), torch.stack(pz_mu, dim=0), torch.stack(
            pz_log_sigma,
            dim=0)
        # P(m|z,u)
        m_tm1 = cuda_(Variable(torch.ones(1, batch_size).long()))  # GO token
        pm_dec_proba, m_dec_outs = [],[]

        turn_states['pv_z_dec_outs'], turn_states['pv_pz_proba'] = pz_dec_outs, pz_proba

        if is_train or test_type=='post':
            m_length = m_input.size(0)  # Tm
            for t in range(m_length):
                teacher_forcing = toss_(self.teacher_force)
                proba, last_hidden, dec_out = self.m_decoder(pz_dec_outs, pz_proba, u_enc_out, m_tm1, degree_input, last_hidden)
                if teacher_forcing:
                    m_tm1 = m_input[t].view(1, -1)
                else:
                    _, m_tm1 = torch.topk(proba, 1)
                    m_tm1 = m_tm1.view(1, -1)
                pm_dec_proba.append(proba)
                m_dec_outs.append(dec_out)

            pm_dec_proba = torch.stack(pm_dec_proba, dim=0)  # [T,B,V]

            # Q(z|u,m)

            p_enc_out, p_enc_hidden = self.m_encoder(m_input, m_len)
            last_hidden = p_enc_hidden[:-1]

            qz_ae = cuda_(Variable(torch.zeros(1, batch_size, self.embed_size)))
            qz_proba, qz_mu, qz_log_sigma = [], [], []
            for t in range(z_length):
                if cfg.sampling:
                    rand_eps = cfg.alpha * Variable(torch.normal(means=torch.zeros(1, batch_size, cfg.embedding_size), std=1))
                else:
                    rand_eps = Variable(torch.zeros(1, batch_size, cfg.embedding_size))
                if cfg.cuda: rand_eps = rand_eps.cuda()
                qz_ae, gru_out, last_hidden, proba, appr_emb, log_sigma_ae = \
                    self.qz_decoder(u_input=u_input, u_enc_out=u_enc_out, pv_pz_proba=pv_pz_proba,
                                    pv_z_dec_out=pv_z_outs,
                                    m_input=p_input, m_enc_out=p_enc_out, u_input_np=u_input_np, m_input_np=p_input_np,
                                    embed_z=qz_ae, last_hidden=last_hidden, rand_eps=rand_eps)
                qz_proba.append(proba)
                qz_mu.append(appr_emb)
                qz_log_sigma.append(log_sigma_ae)
            qz_proba, qz_mu, qz_log_sigma = torch.stack(qz_proba, dim=0), torch.stack(qz_mu, dim=0), torch.stack(
                qz_log_sigma,
                dim=0)
            if is_train:
                return pz_proba, qz_proba, pm_dec_proba, pz_mu, pz_log_sigma, qz_mu, qz_log_sigma, turn_states
            else:
                qz_index = self.pz_max_sampling(qz_proba)
                return None, qz_index, turn_states
        else:
            if not self.beam_search:
                m_output_index = self.greedy_decode(pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden, degree_input)
            else:
                m_output_index = self.beam_search_decode(pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden,
                                                         degree_input,
                                                         self.eos_token_idx)
            pz_index = self.pz_max_sampling(pz_proba)
            return m_output_index, pz_index, turn_states

    def greedy_decode(self, pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden, degree_input):
        """
        greedy decoding of the response
        :param pz_dec_outs:
        :param u_enc_out:
        :param m_tm1:
        :param last_hidden:
        :return: nested-list
        """
        decoded = []
        for t in range(self.max_ts):
            proba, last_hidden, _ = self.m_decoder(pz_dec_outs, pz_proba, u_enc_out, m_tm1, degree_input, last_hidden)
            mt_proba, mt_index = torch.topk(proba, 1)  # [B,1]
            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index)
            m_tm1 = cuda_(Variable(mt_index).view(1, -1))
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        return [list(_) for _ in decoded]

    def pz_max_sampling(self, pz_proba):
        """
        Max-sampling procedure of pz during testing.
        :param pz_proba: # [Tz, B, Vz]
        :return: nested-list: B * [T]
        """
        pz_proba = pz_proba.data
        z_proba, z_token = torch.topk(pz_proba, 1, dim=2)  # [Tz, B, 1]
        z_token = list(z_token.squeeze(2).transpose(0, 1))
        return [list(_) for _ in z_token]

    def pz_selective_sampling(self, pz_proba):
        """
        Selective sampling of pz
        """
        if cfg.spv_proportion == 0:
            return self.pz_max_sampling(pz_proba)
        pz_proba = pz_proba.data
        z_proba, z_token = torch.topk(pz_proba, pz_proba.size(0), dim=2)
        z_token = z_token.transpose(0, 1)  # [B,Tz,top_Tz]
        all_sampled_z = []
        for b in range(z_token.size(0)):
            sampled_z = []
            for t in range(z_token.size(1)):
                for i in range(z_token.size(2)):
                    if z_token[b][t][i] not in sampled_z:
                        sampled_z.append(z_token[b][t][i])
                        break
            all_sampled_z.append(sampled_z)
        return all_sampled_z

    def beam_search_decode_single(self, pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden, degree_input,
                                  eos_token_id):
        """
        Single beam search decoding. Batch size have to be 1.
        :param eos_token_id:
        :param degree_input:
        :param last_hidden:
        :param m_tm1:
        :param pz_dec_outs: [T,1,H]
        :param pz_proba: [T,1,V]
        :param u_enc_out: [T,1,H]
        :return:
        """
        eos_token_id = self.vocab.encode(cfg.eos_m_token)
        batch_size = pz_dec_outs.size(1)
        if batch_size != 1:
            raise ValueError('"Beam search single" requires batch size to be 1')

        class BeamState:
            def __init__(self, score, last_hidden, decoded, length):
                """
                Beam state in beam decoding
                :param score: sum of log-probabilities
                :param last_hidden: last hidden
                :param decoded: list of *Variable[1*1]* of all decoded words
                :param length: current decoded sentence length
                """
                self.score = score
                self.last_hidden = last_hidden
                self.decoded = decoded
                self.length = length

            def update_clone(self, score_incre, last_hidden, decoded_t):
                decoded = copy.copy(self.decoded)
                decoded.append(decoded_t)
                clone = BeamState(self.score + score_incre, last_hidden, decoded, self.length + 1)
                return clone

        def beam_result_valid(decoded_t):
            pz_max_samples = self.pz_selective_sampling(pz_proba)
            requested, start = [], False
            t = 0
            while t < len(pz_max_samples[0]) and pz_max_samples[0][t] != self.vocab.encode('EOS_Z1'):
                t += 1
            t += 1
            while t < len(pz_max_samples[0]) and pz_max_samples[0][t] != self.vocab.encode('EOS_Z2'):
                requested.append(self.vocab.decode(pz_max_samples[0][t]))
                t += 1
            decoded_t = [_.view(-1).data[0] for _ in decoded_t]
            decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)

            requested = set(requested).intersection(['address', 'food', 'pricerange', 'phone', 'postcode'])
            # return True
            for rq in requested:
                if '%s SLOT' % rq not in decoded_sentence:
                    #print('Fail %s' % decoded_sentence)
                    return False
            #print('Success %s' % decoded_sentence)
            return True

        def score_bonus(state, decoded):
            """
            bonus scheme: bonus per token, or per new decoded slot.
            :param state:
            :return:
            """
            bonus = cfg.beam_len_bonus
            decoded = self.vocab.decode(decoded)
            decoded_t = [_.view(-1).data[0] for _ in state.decoded]
            decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)
            decoded_sentence = decoded_sentence.split()
            if len(decoded_sentence) >= 1 and decoded_sentence[-1] == decoded: # repeated words
                bonus -= 10000
            if decoded == '**unknown**':
                bonus -= 3.0
            return bonus

        def soft_score_incre(score, turn):
            return score

        finished, failed = [], []
        states = []  # sorted by score decreasingly
        dead_k = 0
        states.append(BeamState(0, last_hidden, [m_tm1], 0))
        for t in range(self.max_ts):
            new_states = []
            k = 0
            while k < len(states) and k < self.beam_size - dead_k:
                state = states[k]
                last_hidden, m_tm1 = state.last_hidden, state.decoded[-1]
                proba, last_hidden = self.m_decoder(pz_dec_outs, pz_proba, u_enc_out, m_tm1, degree_input, last_hidden)
                proba = torch.log(proba)
                mt_proba, mt_index = torch.topk(proba, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = soft_score_incre(mt_proba[0][new_k].data[0], t) + score_bonus(state, mt_index[0][new_k].data[0])
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = mt_index[0][new_k]
                    if self.vocab.decode(decoded_t.data[0]) == cfg.eos_m_token:
                        if beam_result_valid(state.decoded):
                            finished.append(state)
                            dead_k += 1
                        else:
                            failed.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)
                        #beam_result_valid(new_state.decoded)
                        #print(self.vocab.decode(decoded_t.view(-1).data[0]), t, new_k)
                k += 1
            if self.beam_size - dead_k < 0:
                break
            new_states = new_states[:self.beam_size - dead_k]
            new_states.sort(key=lambda x: -x.score)
            states = new_states

            if t == self.max_ts - 1 and not finished:
                finished = failed
                if not finished:
                    finished.append(states[0])

        finished.sort(key=lambda x: -x.score)
        decoded_t = finished[0].decoded
        decoded_t = [_.view(-1).data[0] for _ in decoded_t]
        decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)
        print(decoded_sentence) 
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        return generated

    def beam_search_decode(self, pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden, degree_input, eos_token_id):
        vars = torch.split(pz_dec_outs, 1, dim=1), torch.split(pz_proba, 1, dim=1), torch.split(u_enc_out, 1,
                                                                                                dim=1), torch.split(
            m_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1), torch.split(degree_input, 1, dim=0)
        decoded = []
        for pz_dec_out_s, pz_proba_s, u_enc_out_s, m_tm1_s, last_hidden_s, degree_input_s in zip(*vars):
            decoded_s = self.beam_search_decode_single(pz_dec_out_s, pz_proba_s, u_enc_out_s, m_tm1_s, last_hidden_s,
                                                       degree_input_s, eos_token_id)
            decoded.append(decoded_s)
        return [list(_.view(-1)) for _ in decoded]

    def supervised_loss(self, pz_proba, qz_proba, pm_dec_proba, z_input, m_input):
        pr_loss = self.pr_loss(pz_proba.view(-1, pz_proba.size(2)), z_input.view(-1))
        m_loss = self.dec_loss(pm_dec_proba.view(-1, pm_dec_proba.size(2)), m_input.view(-1))
        q_loss = self.q_loss(qz_proba.view(-1, pz_proba.size(2)), z_input.view(-1))
        if cfg.pretrain:
            loss = q_loss
        else:
            loss = pr_loss + m_loss + q_loss
            #loss=pr_loss
        return loss, pr_loss, m_loss, q_loss

    def unsupervised_loss(self, mu_q, log_sigma_q, mu_p, log_sigma_p, pm_dec_proba, m_input, pz_proba, qz_proba):
        m_loss = self.dec_loss(pm_dec_proba.view(-1, pm_dec_proba.size(2)), m_input.view(-1))
        kl_div_loss = self.kl_loss(pz_proba, qz_proba)
        loss = m_loss + self.alpha * kl_div_loss
        return loss, m_loss, self.alpha * kl_div_loss

    def self_adjust(self, epoch):
        pass
