import torch
import random
import numpy as np
from config import global_config as cfg
from reader import CamRest676Reader, get_glove_matrix
from reader import KvretReader
from reader import UbuntuDialogueReader
from reader import JDCorpusReader
from unsup_net import UnsupervisedSEDST, cuda_
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from reader import pad_sequences
import argparse, time
from metric import CamRestEvaluator, KvretEvaluator, GenericEvaluator
import logging


class Model:
    def __init__(self, dataset, inference_only=False):
        reader_dict = {
            'camrest': CamRest676Reader,
            'kvret': KvretReader,
            'ubuntu': UbuntuDialogueReader,
            'jd': JDCorpusReader
        }
        model_dict = {
            'SEDST': UnsupervisedSEDST,
        }
        evaluator_dict = {
            'camrest': CamRestEvaluator,
            'kvret': KvretEvaluator,
            'ubuntu': GenericEvaluator,
            'jd': GenericEvaluator
        }
        self.reader = reader_dict[dataset]()
        self.m = model_dict[cfg.m](embed_size=cfg.embedding_size,
                                   hidden_size=cfg.hidden_size,
                                   q_hidden_size=cfg.q_hidden_size,
                                   vocab_size=cfg.vocab_size,
                                   layer_num=cfg.layer_num,
                                   dropout_rate=cfg.dropout_rate,
                                   z_length=cfg.z_length,
                                   alpha=cfg.alpha,
                                   max_ts=cfg.max_ts,
                                   beam_search=cfg.beam_search,
                                   beam_size=cfg.beam_size,
                                   eos_token_idx=self.reader.vocab.encode('EOS_M'),
                                   vocab=self.reader.vocab,
                                   teacher_force=cfg.teacher_force,
                                   degree_size=cfg.degree_size)
        self.EV = evaluator_dict[dataset]  # evaluator class
        if cfg.cuda: self.m = self.m.cuda()
        self.base_epoch = -1

    def _convert_batch(self, py_batch, prev_z_py=None):
        u_input_py = py_batch['user']
        u_len_py = py_batch['u_len']
        kw_ret = {}
        if cfg.prev_z_method == 'concat' and prev_z_py is not None:
            for i in range(len(u_input_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    u_input_py[i] = prev_z_py[i][:idx + 1] + u_input_py[i]
                else:
                    u_input_py[i] = prev_z_py[i] + u_input_py[i]
                u_len_py[i] = len(u_input_py[i])
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2  # unk
        elif cfg.prev_z_method == 'separate' and prev_z_py is not None:
            for i in range(len(prev_z_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    prev_z_py[i] = prev_z_py[i][:idx + 1]
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2  # unk
            prev_z_input_np = pad_sequences(prev_z_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in prev_z_py])
            prev_z_input = cuda_(Variable(torch.from_numpy(prev_z_input_np).long()))
            kw_ret['prev_z_len'] = prev_z_len
            kw_ret['prev_z_input'] = prev_z_input
            kw_ret['prev_z_input_np'] = prev_z_input_np

        degree_input_np = np.array(py_batch['degree'])
        u_input_np = pad_sequences(u_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        z_input_np = pad_sequences(py_batch['latent'], padding='post').transpose((1, 0))
        if cfg.pretrain:
            m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose(
                (1, 0))
        else:
            m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='pre').transpose(
                (1, 0))
        p_input_np = pad_sequences(py_batch['post'], cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        u_len = np.array(u_len_py)
        m_len = np.array(py_batch['m_len'])
        p_len = np.array(py_batch['p_len'])
        degree_input = cuda_(Variable(torch.from_numpy(degree_input_np).float()))
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        z_input = cuda_(Variable(torch.from_numpy(z_input_np).long()))
        m_input = cuda_(Variable(torch.from_numpy(m_input_np).long()))
        p_input = cuda_(Variable(torch.from_numpy(p_input_np).long()))
        supervised = py_batch['supervised'][0]
        kw_ret['z_input_np'] = z_input_np
        return u_input, u_input_np, z_input, m_input, m_input_np, p_input, p_input_np, u_len, m_len, p_len, \
               degree_input, supervised, kw_ret

    def train(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        train_time = 0
        for epoch in range(cfg.epoch_num):
            sw = time.time()
            if epoch < cfg.base_epoch:
                continue
            sup_loss, unsup_loss = 0, 0
            sup_cnt, unsup_cnt = 0, 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=1e-6)
            for iter_num, dial_batch in enumerate(data_iterator):
                if epoch == cfg.base_epoch and iter_num < cfg.base_iter:
                    continue
                turn_states = {}
                turn_states_q = {}
                prev_z = None
                trunc_cnt = 1
                for turn_num, turn_batch in enumerate(dial_batch):
                    if cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()
                    u_input, u_input_np, z_input, m_input, m_input_np, p_input, p_input_np, u_len, \
                    m_len, p_len, degree_input, supervised, kw_ret \
                        = self._convert_batch(turn_batch, prev_z)

                    loss, m_loss, p_loss, kl_div_loss, turn_states, turn_states_q = self.m(u_input=u_input,
                                                                                           z_input=None,
                                                                                           m_input=m_input,
                                                                                           p_len=p_len,
                                                                                           degree_input=degree_input,
                                                                                           u_input_np=u_input_np,
                                                                                           m_input_np=m_input_np,
                                                                                           z_supervised=False,
                                                                                           turn_states=turn_states,
                                                                                           p_input=p_input,
                                                                                           p_input_np=p_input_np,
                                                                                           u_len=u_len, m_len=m_len,
                                                                                           mode='train',
                                                                                           turn_states_q=turn_states_q,
                                                                                           **kw_ret)
                    if turn_num == len(dial_batch) - 1 or (trunc_cnt and trunc_cnt % cfg.trunc_turn == 0):
                        for k in turn_states:
                            turn_states[k] = cuda_(Variable(turn_states[k].data))
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    trunc_cnt += 1
                    grad = torch.nn.utils.clip_grad_norm(self.m.parameters(), 4.0)
                    optim.step()
                    unsup_loss += loss.item()
                    if cfg.truncated and not np.isnan(loss.data.cpu().numpy()) and not np.isnan(
                            grad) and iter_num % 10 == 0 and iter_num != 0:
                        self.save_model(epoch)
                    unsup_cnt += 1
                    logging.debug(
                        'unsupervised loss:{} m_loss:{} p_loss:{} kl_div_loss:{} grad:{}'.format(loss.item(),
                                                                                                 m_loss.item(),
                                                                                                 p_loss.item(),
                                                                                                 kl_div_loss.data[
                                                                                                     0], grad))
                    prev_z = turn_batch['latent']

            epoch_sup_loss, epoch_unsup_loss = sup_loss / (sup_cnt + 1e-8), unsup_loss / (unsup_cnt + 1e-8)
            train_time += time.time() - sw

            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%6f unsup:%6f' % (epoch, epoch_sup_loss, epoch_unsup_loss))
            # do validation
            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%6f unsup:%6f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %6f' % (epoch, time.time() - sw))
            valid_loss = valid_sup_loss + valid_unsup_loss
            self.save_model(epoch)
            if valid_loss <= prev_min_loss:
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %6f' % (early_stop_count, lr))

    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        with torch.no_grad():
            data_iterator = self.reader.mini_batch_iterator(data)
            mode = 'test'  # if not cfg.pretrain else 'pretrain_test'
            for batch_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                turn_states_q = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    u_input, u_input_np, z_input, m_input, m_input_np, p_input, p_input_np, u_len, \
                    m_len, p_len, degree_input, supervised, kw_ret \
                        = self._convert_batch(turn_batch, prev_z)
                    m_idx, z_idx, turn_states = self.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                                       m_input=m_input,
                                                       degree_input=degree_input, u_input_np=u_input_np,
                                                       m_input_np=m_input_np,
                                                       p_input=p_input, p_input_np=p_input_np, p_len=p_len,
                                                       m_len=m_len, z_supervised=None, turn_states=turn_states,
                                                       **kw_ret)
                    if not cfg.last_turn_only or turn_num == len(dial_batch) - 1:
                        self.reader.wrap_result(turn_batch, m_idx, z_idx)
                    prev_z = z_idx
                # print('{}\r'.format(batch_num))
            ev = self.EV(result_path=cfg.result_path)
            res = ev.run_metrics()
        self.m.train()
        return res

    def validate(self, data='dev'):
        self.m.eval()
        with torch.no_grad():
            data_iterator = self.reader.mini_batch_iterator(data)
            sup_loss, unsup_loss = 0, 0
            sup_cnt, unsup_cnt = 0, 0
            for d, dial_batch in enumerate(data_iterator):
                turn_states = {}
                for turn_num, turn_batch in enumerate(dial_batch):
                    if turn_num <= 0 or turn_num < len(dial_batch) - cfg.max_turn:
                        continue
                    u_input, u_input_np, z_input, m_input, m_input_np, p_input, p_input_np, u_len, \
                    m_len, p_len, degree_input, supervised, kw_ret \
                        = self._convert_batch(turn_batch)

                    loss, m_loss, p_loss, kl_div_loss, turn_states, _ = self.m(u_input=u_input, z_input=None,
                                                                               m_input=m_input,
                                                                               z_supervised=False,
                                                                               turn_states=turn_states,
                                                                               u_input_np=u_input_np,
                                                                               m_input_np=m_input_np,
                                                                               p_input=p_input, p_input_np=p_input_np,
                                                                               p_len=p_len,
                                                                               u_len=u_len, m_len=m_len, mode='train',
                                                                               degree_input=degree_input,
                                                                               turn_states_q={}, **kw_ret)
                    if not cfg.last_turn_only or turn_num == len(dial_batch) - 1:
                        unsup_loss += m_loss.item()
                        unsup_cnt += 1
                    logging.debug(
                        'unsupervised loss:{} m_loss:{} p_loss:{} kl_div_loss:{}'.format(loss.item(), m_loss.item(),
                                                                                         p_loss.item(),
                                                                                         kl_div_loss.item()))
                    for k in turn_states:
                        turn_states[k] = turn_states[k].detach()

            sup_loss /= (sup_cnt + 1e-8)
            unsup_loss /= (unsup_cnt + 1e-8)
        self.m.train()
        res = self.eval()
        return sup_loss, unsup_loss

    def save_model(self, epoch, path=None):
        if not path:
            path = cfg.model_path
        all_state = {'sedst': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        with open(path, 'wb') as f:
            torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        with open(path, 'rb') as f:
            all_state = torch.load(path)
        self.m.load_state_dict(all_state['sedst'], strict=False)
        self.base_epoch = all_state.get('epoch', 0)


    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        initial_arr = self.m.u_encoder.embedding.weight.data.cpu().numpy()
        mat = get_glove_matrix(self.reader.vocab, initial_arr)
        # np.save('./data/embedding.npy',mat)
        # mat = np.load('./data/embedding.npy')
        embedding_arr = torch.from_numpy(mat)

        self.m.u_encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.p_encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.m_decoder.emb.weight.data.copy_(embedding_arr)
        self.m.p_decoder.emb.weight.data.copy_(embedding_arr)
        self.m.qz_decoder.mu.weight.data.copy_(embedding_arr.transpose(1, 0))
        self.m.pz_decoder.mu.weight.data.copy_(embedding_arr.transpose(1, 0))
        if freeze:
            self.freeze_module(self.m.u_encoder.embedding)
            self.freeze_module(self.m.m_e.embedding)
            self.freeze_module(self.m.m_decoder.emb)

    def count_params(self):

        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters])

        print('total trainable params: %d' % param_cnt)


def main():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-dataset')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.init_handler(args.dataset)

    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)

    logging.debug(str(cfg))
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.debug('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = args.mode
    m = Model(args.dataset.split('-')[-1])
    m.count_params()
    if args.mode == 'train':
        m.load_glove_embedding()
        m.train()
    elif args.mode == 'adjust':
        m.load_model()
        m.train()
    elif args.mode == 'test':
        m.load_model()
        m.eval()


if __name__ == '__main__':
    main()
