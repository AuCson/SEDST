import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
from config import global_config as cfg
import copy, random, time, logging
from nltk.corpus import stopwords
import re
from rnn_net import *

sw = set(stopwords.words())
sw = sw.union({',', '.', '?', '!', '"', "'", ':', ';', '(', ')',
               '...', '**unknown**', '<unk>', '<go>', '<pad>', '<go2>', '__eot__', 'EOS_M',
               '__eou__', '</s>'})
sw_index = set()


def toss_(p):
    return random.randint(0, 99) <= p


def orth_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + gru.hidden_size], gain=1)
    return gru


def get_sparse_input_efficient(x_input_np):
    ignore_index = sw_index
    result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size), dtype=np.float32)
    result.fill(0.)
    for t in range(x_input_np.shape[0]):
        for b in range(x_input_np.shape[1]):
            if x_input_np[t][b] not in ignore_index:
                result[t][b][x_input_np[t][b]] = 1.0
    result_np = result.transpose((1, 0, 2))
    result = torch.from_numpy(result_np).float()
    return result


def mask_prob(score, prob, aux=None):
    """

    :param score: [B,T]
    :param prob: [B,T,V]
    :param aux:
    :return:
    """
    score = score.contiguous()
    prob = prob.contiguous()
    score = cuda_(score, aux)
    prob = cuda_(prob, aux)
    res = cuda_(score.unsqueeze(1).bmm(prob).squeeze(1))  # [B, V]
    freq_mask = cuda_(Variable(torch.Tensor([0] * cfg.freq_thres + [1] * (prob.size(2) - cfg.freq_thres))))  # [V]
    return res * freq_mask


def shift(pz_proba):
    first_input = np.zeros((pz_proba.size(1), pz_proba.size(2)))
    first_input.fill(0.)
    first_input = cuda_(Variable(torch.from_numpy(first_input)).float())
    pz_proba = list(pz_proba)[:-1]
    pz_proba.insert(0, first_input)
    pz_proba = torch.stack(pz_proba, 0).contiguous()
    return pz_proba


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, input_seqs, hidden=None):
        embedded = self.embedding(input_seqs)
        outputs, hidden = self.gru(embedded, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class TextSpanDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate):
        super().__init__()
        self.attn_u = Attn(hidden_size)
        self.attn_z = Attn(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, dropout=dropout_rate)
        self.ln1 = LayerNormalization(hidden_size)

        self.w1 = nn.Linear(hidden_size, vocab_size)
        self.proj_copy1 = nn.Linear(hidden_size * 2, hidden_size)
        self.v1 = nn.Linear(hidden_size, 1)

        self.proj_copy2 = nn.Linear(hidden_size * 2, hidden_size)
        self.v2 = nn.Linear(hidden_size, 1)
        self.mu = nn.Linear(vocab_size, embed_size)
        self.dropout_rate = dropout_rate

        self.gru = orth_gru(self.gru)

        self.copy_weight = 1

    def forward(self, u_input, u_enc_out, pv_pz_proba, pv_z_dec_out, embed_z, last_hidden, u_input_np,
                m_input_np, sparse_u_input):
        u_context = self.attn_u(last_hidden, u_enc_out)
        embed_z = F.dropout(embed_z, self.dropout_rate)
        gru_in = torch.cat([u_context, embed_z], 2)
        gru_out, last_hidden = self.gru(gru_in, last_hidden)

        gru_out = self.ln1(gru_out)

        gen_score = self.w1(gru_out).squeeze(0)
        max_len = u_enc_out.size(0)
        u_copy_score = F.tanh(self.proj_copy1(torch.cat([u_enc_out, gru_out.repeat(max_len, 1, 1)], 2)))  # [T,B,H]
        u_copy_score = self.v1(u_copy_score).squeeze(2).transpose(0, 1)  # [B,T]

        if pv_pz_proba is not None:
            pv_pz_proba = shift(pv_pz_proba)
            pv_z_copy_score = F.tanh(
                self.proj_copy2(torch.cat([pv_z_dec_out, gru_out.repeat(pv_z_dec_out.size(0), 1, 1)], 2)))  # [T,B,H]
            pv_z_copy_score = self.v2(pv_z_copy_score).squeeze(2).transpose(0, 1)  # [B,T]
            scores = F.softmax(torch.cat([gen_score, u_copy_score, pv_z_copy_score], dim=1), dim=1)
            cum_idx = [gen_score.size(1), u_copy_score.size(1), pv_z_copy_score.size(1)]
            for i in range(len(cum_idx) - 1):
                cum_idx[i + 1] += cum_idx[i]
            cum_idx.insert(0, 0)
            gen_score, u_copy_score, pv_z_copy_score = tuple(
                [scores[:, cum_idx[i]:cum_idx[i + 1]] for i in range(3)])
            u_copy_score = mask_prob(u_copy_score, sparse_u_input, aux=cfg.aux_device)
            pv_z_copy_score = mask_prob(pv_z_copy_score, pv_pz_proba.transpose(0, 1), aux=cfg.aux_device)
            proba = gen_score + self.copy_weight * u_copy_score + self.copy_weight * pv_z_copy_score
        else:
            scores = F.softmax(torch.cat([gen_score, u_copy_score], dim=1), dim=1)
            cum_idx = [gen_score.size(1), u_copy_score.size(1)]
            for i in range(len(cum_idx) - 1):
                cum_idx[i + 1] += cum_idx[i]
            cum_idx.insert(0, 0)
            gen_score, u_copy_score = tuple(
                [scores[:, cum_idx[i]:cum_idx[i + 1]] for i in range(2)])
            u_copy_score = mask_prob(u_copy_score, sparse_u_input, aux=cfg.aux_device)
            proba = gen_score + self.copy_weight * u_copy_score

        mu_ae = self.mu(proba)
        return mu_ae.unsqueeze(0), gru_out, last_hidden, proba, mu_ae


class ResponseDecoder(nn.Module):
    """
    Response decoder: P_theta(m_t|s_t, z_t) <- P_theta(m_ti|s_t, z_t, m_t[1..i-1])
    This is a deterministic decoder.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate, flag_size=5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.attn_z = Attn(hidden_size)
        self.attn_u = Attn(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size * 2, hidden_size, dropout=dropout_rate)
        self.gru = orth_gru(self.gru)
        self.ln1 = LayerNormalization(hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.proj_copy1 = nn.Linear(hidden_size * 2, hidden_size)
        self.v1 = nn.Linear(hidden_size, 1)
        # self.proj_copy2 = nn.Linear(hidden_size,1)
        self.dropout_rate = dropout_rate
        # orth_gru(self.gru)
        self.copy_weight = 1

    def forward(self, z_enc_out, pz_proba, u_enc_out, m_t_input, last_hidden, flag=False):
        """
        decode the response: P(m|u,z)
        :param pz_proba: [Tz,B,V], output of the prior decoder
        :param z_enc_out: [Tz,B,H]
        :param u_enc_out: [T,B,H]
        :param m_t_input: [1,B]
        :param last_hidden:
        :return: proba: [1,B,V]
        """
        batch_size = z_enc_out.size(1)
        m_embed = self.emb(m_t_input)
        z_context = F.dropout(self.attn_z(last_hidden, z_enc_out), self.dropout_rate)
        u_context = F.dropout(self.attn_u(last_hidden, u_enc_out), self.dropout_rate)
        # d_control = self.w4(z_context) + torch.mul(F.sigmoid(self.gate_z(z_context)), self.w5(u_context))
        gru_out, last_hidden = self.gru(torch.cat([z_context, u_context, m_embed], dim=2),
                                        last_hidden)
        gru_out = self.ln1(gru_out)

        gen_score = self.proj(gru_out).squeeze(0)

        z_copy_score = F.tanh(
            self.proj_copy1(torch.cat([z_enc_out, gru_out.repeat(z_enc_out.size(0), 1, 1)], 2)))  # [T,B,H]
        z_copy_score = self.v1(z_copy_score).squeeze(2).transpose(0, 1)  # [B,T]

        scores = F.softmax(torch.cat([gen_score, z_copy_score], dim=1), dim=1)
        gen_score, z_copy_score = scores[:, :gen_score.size(1)], scores[:, gen_score.size(1):]
        z_copy_score = mask_prob(z_copy_score, pz_proba.transpose(0, 1), aux=cfg.aux_device)
        proba = gen_score + self.copy_weight * z_copy_score  # [B,V]
        return proba, last_hidden


class MultinomialKLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p_proba, q_proba):  # [B, T, V]
        loss = q_proba * (torch.log(q_proba) - torch.log(p_proba))
        loss = torch.sum(loss)
        return loss / (p_proba.size(1) * p_proba.size(0))


class UnsupervisedSEDST(nn.Module):
    def __init__(self, embed_size, hidden_size, q_hidden_size, vocab_size, layer_num, dropout_rate,
                 z_length, alpha,
                 max_ts, beam_search=False, teacher_force=100, **kwargs):
        super().__init__()
        self.u_encoder = DynamicEncoder(vocab_size, embed_size, hidden_size, layer_num, dropout_rate)
        self.p_encoder = DynamicEncoder(vocab_size, embed_size, q_hidden_size, layer_num, dropout_rate)

        self.qz_decoder = TextSpanDecoder(embed_size, q_hidden_size, vocab_size, dropout_rate)  # posterior
        self.pz_decoder = TextSpanDecoder(embed_size, hidden_size, vocab_size, dropout_rate)  # prior

        self.m_decoder = ResponseDecoder(embed_size, hidden_size, vocab_size, dropout_rate)
        self.p_decoder = ResponseDecoder(embed_size, q_hidden_size, vocab_size, dropout_rate)

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
                mode, p_input, p_len, p_input_np, **kwargs):
        if mode == 'train' or mode == 'valid':
            if not z_supervised:
                z_input = None
            if z_supervised:
                qz_proba = None
                pz_proba, pm_dec_proba, turn_states = \
                    self.forward_turn(u_input, u_len, m_input=m_input, m_len=m_len, z_input=z_input, is_train=True,
                                      turn_states=turn_states, u_input_np=u_input_np,
                                      m_input_np=m_input_np)

                loss, pr_loss, m_loss, q_loss = self.supervised_loss(torch.log(pz_proba), torch.log(qz_proba),
                                                                     torch.log(pm_dec_proba), z_input, m_input)
                return loss, pr_loss, m_loss, q_loss, turn_states, None
            else:
                # turn states: previous decoded text span S_t for prior network
                # turn states_q: previous decoded text span for posterior network
                turn_states_q = kwargs['turn_states_q']
                pz_proba, pm_dec_proba, turn_states = \
                    self.forward_turn(u_input, u_len, m_input=m_input, m_len=m_len, z_input=z_input, is_train=True,
                                      turn_states=turn_states, u_input_np=u_input_np,
                                      m_input_np=m_input_np)

                qz_proba, qp_dec_proba, turn_states_q = \
                    self.forward_turn(p_input, p_len, m_input=p_input, m_len=p_len, z_input=z_input, is_train=True,
                                      turn_states=turn_states_q, u_input_np=p_input_np,
                                      m_input_np=p_input_np, flag=True)
                for k in turn_states_q:
                    turn_states_q[k] = cuda_(Variable(turn_states_q[k].data))

                loss, m_loss, p_loss, kl_div_loss = self.unsupervised_loss(pz_proba, qz_proba, torch.log(pm_dec_proba),
                                                                           m_input, torch.log(qp_dec_proba), p_input)
            return loss, m_loss, p_loss, kl_div_loss, turn_states, turn_states_q
        elif mode == 'test':
            m_output_index, pz_index, turn_states = self.forward_turn(u_input, u_len=u_len, is_train=False,
                                                                      turn_states=turn_states,
                                                                      u_input_np=u_input_np, m_input_np=m_input_np,
                                                                      flag=cfg.pretrain,
                                                                      last_turn=kwargs.get('last_turn', False))
            return m_output_index, pz_index, turn_states

    def forward_turn(self, u_input, u_len, turn_states, is_train, u_input_np, m_input_np=None,
                     m_input=None, m_len=None, z_input=None, flag=False, last_turn=False):
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

        decoder = self.m_decoder if not flag else self.p_decoder
        z_decoder = self.pz_decoder if not flag else self.qz_decoder
        encoder = self.u_encoder if not flag else self.p_encoder

        pv_pz_proba = turn_states.get('pv_pz_proba', None)
        pv_z_outs = turn_states.get('pv_z_dec_outs', None)

        batch_size = u_input.size(1)
        u_enc_out, u_enc_hidden = encoder(u_input, u_len)
        last_hidden = u_enc_hidden[:-1]
        # initial approximate embedding: SOS token initialized with all zero
        # Pi(z|u)
        pz_ae = cuda_(Variable(torch.zeros(1, batch_size, self.embed_size)))
        pz_proba, pz_mu = [], []
        pz_dec_outs = []
        z_length = z_input.size(0) if z_input is not None else self.z_length
        sparse_u_input = Variable(get_sparse_input_efficient(u_input_np), requires_grad=False)

        for t in range(z_length):
            pz_ae, last_hidden, pz_dec_out, proba, mu_ae = \
                z_decoder(u_input=u_input, u_enc_out=u_enc_out, pv_pz_proba=pv_pz_proba, pv_z_dec_out=pv_z_outs,
                          embed_z=pz_ae, last_hidden=last_hidden, u_input_np=u_input_np,
                          m_input_np=m_input_np, sparse_u_input=sparse_u_input)
            pz_proba.append(proba)
            pz_mu.append(mu_ae)
            pz_dec_outs.append(pz_dec_out)
        pz_dec_outs = torch.cat(pz_dec_outs, dim=0)  # [Tz,B,H]
        pz_proba, pz_mu = torch.stack(pz_proba, dim=0), torch.stack(pz_mu, dim=0)
        shift_pz_proba = shift(pz_proba)
        # P(m|z,u)
        m_tm1 = cuda_(Variable(torch.ones(1, batch_size).long()))  # GO token
        pm_dec_proba = []

        turn_states = {
            'pv_z_dec_outs': pz_dec_outs,
            'pv_pz_proba': pz_proba,
        }
        if flag:
            last_hidden = u_enc_hidden[-1:]  # backward pass
        else:
            last_hidden = u_enc_hidden[:-1]  # forward pass
        if is_train:
            m_length = m_input.size(0)  # Tm
            for t in range(m_length):
                teacher_forcing = toss_(self.teacher_force)
                proba, last_hidden = decoder(pz_dec_outs, shift_pz_proba, u_enc_out, m_tm1, last_hidden,
                                             flag)
                if teacher_forcing:
                    m_tm1 = m_input[t].view(1, -1)
                else:
                    _, m_tm1 = torch.topk(proba, 1)
                    m_tm1 = m_tm1.view(1, -1)
                pm_dec_proba.append(proba)

            pm_dec_proba = torch.stack(pm_dec_proba, dim=0)  # [T,B,V]
            return pz_proba, pm_dec_proba, turn_states
        else:
            if last_turn or not cfg.last_turn_only:
                if not self.beam_search:
                    m_output_index = self.greedy_decode(pz_dec_outs, shift_pz_proba, u_enc_out, m_tm1, last_hidden,
                                                        flag)
                else:
                    m_output_index = self.beam_search_decode(pz_dec_outs, shift_pz_proba, u_enc_out, m_tm1, last_hidden,
                                                             self.eos_token_idx, flag)
            else:
                m_output_index = None
            pz_index = self.pz_selective_sampling(pz_proba)
            return m_output_index, pz_index, turn_states

    def greedy_decode(self, pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden, flag):
        """
        greedy decoding of the response
        :param pz_dec_outs:
        :param u_enc_out:
        :param m_tm1:
        :param last_hidden:
        :return: nested-list
        """
        decoded = []
        decoder = self.m_decoder if not flag else self.p_decoder
        for t in range(self.max_ts):
            proba, last_hidden = decoder(pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden)
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
        Selective sampling of pz(do max-sampling but prevent repeated words)
        """
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

    def beam_search_decode_single(self, pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden,
                                  eos_token_id, flag):
        """
        Single beam search decoding. Batch size have to be 1.
        :param eos_token_id:
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
            decoded_t = [_.item() for _ in decoded_t]
            decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)
            # return True
            return len(decoded_sentence.split(' ')) >= 5 and '[' not in decoded_sentence

        def score_bonus(state, decoded, dead_k, t):
            """
            bonus scheme: bonus per token, or per new decoded slot.
            :param state:
            :return:
            """
            bonus = cfg.beam_len_bonus
            decoded = self.vocab.decode(decoded)
            decoded_t = [_.item() for _ in state.decoded]
            decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)
            decoded_sentence = decoded_sentence.split()
            if len(decoded_sentence) >= 1 and decoded_sentence[-1] == decoded:
                bonus -= 10000
            if decoded == '**unknown**' or decoded == '<unk>':
                bonus -= 3.0

            bonus -= self.repeat_penalty(decoded_sentence)
            return bonus

        def soft_score_incre(score, turn):
            return score

        decoder = self.m_decoder if not flag else self.p_decoder
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
                proba, last_hidden = decoder(pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden)
                proba = torch.log(proba)
                mt_proba, mt_index = torch.topk(proba, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = soft_score_incre(mt_proba[0][new_k].item(), t) + \
                                  score_bonus(state, mt_index[0][new_k].item(), dead_k, t)
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = mt_index[0][new_k]
                    if self.vocab.decode(decoded_t.item()) == cfg.eos_m_token:  # and k == 0:
                        if new_k == 0:
                            if beam_result_valid(state.decoded):
                                finished.append(state)
                                dead_k += 1
                            else:
                                failed.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)
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
        decoded_t = [_.item() for _ in decoded_t]
        decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)
        print(decoded_sentence)
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        return generated

    def beam_search_decode(self, pz_dec_outs, pz_proba, u_enc_out, m_tm1, last_hidden, eos_token_id,
                           flag=False):
        vars = torch.split(pz_dec_outs, 1, dim=1), torch.split(pz_proba, 1, dim=1), torch.split(u_enc_out, 1,
                                                                                                dim=1), torch.split(
            m_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1)
        decoded = []
        for pz_dec_out_s, pz_proba_s, u_enc_out_s, m_tm1_s, last_hidden_s in zip(*vars):
            decoded_s = self.beam_search_decode_single(pz_dec_out_s, pz_proba_s, u_enc_out_s, m_tm1_s, last_hidden_s,
                                                       eos_token_id, flag)
            decoded.append(decoded_s)
        return [list(_.view(-1)) for _ in decoded]

    def supervised_loss(self, pz_proba, qz_proba, pm_dec_proba, z_input, m_input):
        pr_loss = self.pr_loss(pz_proba.view(-1, pz_proba.size(2)), z_input.view(-1))
        m_loss = self.dec_loss(pm_dec_proba.view(-1, pm_dec_proba.size(2)), m_input.view(-1))
        q_loss = self.q_loss(qz_proba.view(-1, pz_proba.size(2)), z_input.view(-1))
        loss = pr_loss + m_loss + q_loss
        return loss, pr_loss, m_loss, q_loss

    def unsupervised_loss(self, pz_proba, qz_proba, log_pm_dec_proba, m_input, log_qp_dec_proba, p_input):
        m_loss = self.dec_loss(log_pm_dec_proba.view(-1, log_pm_dec_proba.size(2)), m_input.view(-1))
        m_loss = cuda_(m_loss)
        p_loss = self.dec_loss(log_qp_dec_proba.view(-1, log_qp_dec_proba.size(2)), p_input.view(-1))
        p_loss = cuda_(p_loss)
        qz_proba = cuda_(Variable(qz_proba.data))  # qz_proba is detached for loss computation
        kl_div_loss = self.kl_loss(pz_proba, qz_proba)
        loss = m_loss + self.alpha * kl_div_loss + p_loss
        return loss, m_loss, p_loss, self.alpha * kl_div_loss

    def basic_loss(self, log_pm_dec_proba, m_input):
        m_loss = self.dec_loss(log_pm_dec_proba.view(-1, log_pm_dec_proba.size(2)), m_input.view(-1))
        return m_loss

    def self_adjust(self, epoch_num, iter_num):
        pass

    def repeat_penalty(self, seq):
        """
        brute force n^2
        :param seq:
        :return:
        """

        def overlap(s, i, j):
            res = 0
            while i < len(s) and j < len(s):
                if s[i] == s[j]:
                    res += 1
                else:
                    break
                i += 1
                j += 1
            return res

        res = -1
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                res = max(res, overlap(seq, i, j))
        if res <= 4:
            return 0
        else:
            return 1.0 * res
