import torch
import random
import numpy as np
from config import global_config as cfg
from reader import CamRest676Reader, get_glove_matrix
from reader import KvretReader
from semi_sup_net import SemiSupervisedSEDST,cuda_
from torch.optim import Adam
from torch.autograd import Variable
from reader import pad_sequences
import argparse
import logging
from metric import CamRestEvaluator


class Model:
    def __init__(self, dataset):
        if dataset == 'camrest':
            self.reader = CamRest676Reader()
        elif dataset == 'kvret':
            self.reader = KvretReader()
        self.sedst = SemiSupervisedSEDST(embed_size=cfg.embedding_size,
                         hidden_size=cfg.hidden_size,
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
        if cfg.cuda: self.sedst = self.sedst.cuda()
        self.base_epoch = -1

    def _convert_batch(self, py_batch):
        degree_input_np = np.array(py_batch['degree'])
        u_input_np = pad_sequences(py_batch['user'], cfg.u_max_ts, padding='post',truncating='pre').transpose((1, 0))
        z_input_np = pad_sequences(py_batch['latent'], padding='post').transpose((1, 0))
        m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose((1, 0))
        p_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose((1, 0))
        u_len = np.array(py_batch['u_len'])
        m_len = np.array(py_batch['m_len'])
        p_len = np.array(py_batch['p_len'])
        degree_input = Variable(torch.from_numpy(degree_input_np).float())
        u_input = Variable(torch.from_numpy(u_input_np).long())
        z_input = Variable(torch.from_numpy(z_input_np).long())
        m_input = Variable(torch.from_numpy(m_input_np).long())
        p_input = Variable(torch.from_numpy(p_input_np).long())
        if cfg.cuda:
            u_input, z_input, m_input, p_input, degree_input = u_input.cuda(), z_input.cuda(), m_input.cuda(),\
                                                               p_input.cuda(), degree_input.cuda()
        supervised = py_batch['supervised'][0]
        return u_input, u_input_np, z_input, m_input, m_input_np, p_input, p_input_np, u_len, m_len, p_len,\
               degree_input, supervised

    def train(self):
        lr = cfg.lr
        if self.base_epoch == -1:
            self.freeze_params()
        prev_min_loss, early_stop_count = 1e9, cfg.early_stop_count
        for epoch in range(cfg.epoch_num):
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            self.sedst.self_adjust(epoch)
            sup_loss, unsup_loss = 0, 0
            sup_cnt, unsup_cnt = 0, 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.sedst.parameters()))
            for iter_num,dial_batch in enumerate(data_iterator):
                all_turn_states= []                
                turn_states = {}
                for turn_num, turn_batch in enumerate(dial_batch):
                    if turn_num == cfg.trunc_turn:
                        break
                    if cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()
                    u_input, u_input_np, z_input, m_input, m_input_np, p_input, p_input_np, u_len, \
                    m_len, p_len, degree_input, supervised \
                        = self._convert_batch(turn_batch)
                    if supervised:
                        loss, pr_loss, m_loss, q_loss, turn_states = self.sedst(u_input=u_input, z_input=z_input, m_input=m_input, p_len=p_len,
                                                                  degree_input=degree_input, u_input_np=u_input_np,m_input_np=m_input_np,
                                                                  z_supervised=True, turn_states=turn_states, p_input=p_input, p_input_np=p_input_np,
                                                                  u_len=u_len, m_len=m_len, mode='train')
                        loss.backward(retain_graph=turn_num!=len(dial_batch)-1)
                        grad = torch.nn.utils.clip_grad_norm(self.sedst.parameters(),5.0)
                        optim.step()
                        sup_loss += loss.item()
                        sup_cnt += 1
                        logging.debug(
                            'supervised loss:{} pr_loss:{} m_loss:{} q_loss:{} grad:{}'.format(loss.item(), pr_loss.item(),
                                                                                       m_loss.item(), q_loss.item(), grad))
                    else:
                        if cfg.skip_unsup:
                            #logging.debug('skipping unsupervised batch')
                            continue
                        loss, m_loss, kl_div_loss, turn_states = self.sedst(u_input=u_input, z_input=None, m_input=m_input, p_len=p_len,
                                                              degree_input=degree_input, u_input_np=u_input_np,m_input_np=m_input_np,
                                                              z_supervised=False, turn_states=turn_states,p_input=p_input, p_input_np=p_input_np,
                                                              u_len=u_len, m_len=m_len, mode='train')
                        loss.backward(retain_graph=turn_num!=len(dial_batch)-1 and not turn_num == cfg.trunc_turn-1)
                        grad = torch.nn.utils.clip_grad_norm(self.sedst.parameters(),4.0)
                        optim.step()
                        unsup_loss += loss.item()
                        if cfg.truncated and not np.isnan(loss.data.cpu().numpy()) and not np.isnan(grad) and iter_num % 10 == 0:
                            self.save_model(epoch)
                        unsup_cnt += 1
                        logging.debug(
                            'unsupervised loss:{} m_loss:{} kl_div_loss:{} grad:{}'.format(loss.item(), m_loss.item(),
                                                                                           kl_div_loss.item(), grad))
            epoch_sup_loss, epoch_unsup_loss = sup_loss / (sup_cnt + 1e-8), unsup_loss / (unsup_cnt + 1e-8)
            logging.info('avg training loss in epoch %d sup:%f unsup:%f' % (epoch, epoch_sup_loss, epoch_unsup_loss))

            # do validation

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            valid_loss = valid_sup_loss + valid_unsup_loss
            if valid_loss <= prev_min_loss:
                self.save_model(epoch)
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def eval(self, data='test'):
        with torch.no_grad():
            self.sedst.eval()
            self.reader.result_file = None
            data_iterator = self.reader.mini_batch_iterator(data)
            mode = 'test'
            for batch_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                for turn_batch in dial_batch:
                    u_input, u_input_np, z_input, m_input, m_input_np, p_input, p_input_np, u_len, \
                    m_len, p_len, degree_input, supervised \
                        = self._convert_batch(turn_batch)
                    m_idx, z_idx, turn_states = self.sedst(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input, m_input=m_input,
                                                          degree_input=degree_input, u_input_np=u_input_np,m_input_np=m_input_np,
                                                          p_input=p_input, p_input_np=p_input_np, p_len=p_len,
                                                          m_len=m_len, z_supervised=None, turn_states=turn_states)
                    self.reader.wrap_result(turn_batch, m_idx, z_idx)
                print('{}\r'.format(batch_num))
        ev = CamRestEvaluator(cfg.result_path)
        ev.run_metrics()
        self.sedst.train()

    def validate(self, data='dev'):
        self.sedst.eval()
        with torch.no_grad():
            data_iterator = self.reader.mini_batch_iterator(data)
            sup_loss, unsup_loss = 0, 0
            sup_cnt, unsup_cnt = 0, 0
            for dial_batch in data_iterator:
                turn_states = {}
                for turn_num, turn_batch in enumerate(dial_batch):
                    u_input, u_input_np, z_input, m_input, m_input_np, p_input, p_input_np, u_len, \
                    m_len, p_len, degree_input, supervised \
                        = self._convert_batch(turn_batch)
                    if supervised:
                        loss, pr_loss, m_loss, q_loss, turn_states = self.sedst(u_input=u_input, z_input=z_input, m_input=m_input,
                                                                  z_supervised=True, turn_states=turn_states,
                                                                  p_input=p_input, p_input_np=p_input_np, p_len=p_len,
                                                                  degree_input=degree_input,u_input_np=u_input_np,m_input_np=m_input_np,
                                                                  u_len=u_len, m_len=m_len, mode='train')
                        sup_loss += loss.item()
                        sup_cnt += 1
                        logging.debug(
                            'supervised loss:{} pr_loss:{} m_loss:{} q_loss:{}'.format(loss.item(), pr_loss.item(),
                                                                                       m_loss.item(), q_loss.item()))
                    else:
                        loss, m_loss, kl_div_loss, turn_states = self.sedst(u_input=u_input, z_input=None, m_input=m_input,
                                                              z_supervised=False, turn_states=turn_states,u_input_np=u_input_np,m_input_np=m_input_np,
                                                              p_input=p_input, p_input_np=p_input_np, p_len=p_len,
                                                              u_len=u_len, m_len=m_len, mode='train', degree_input=degree_input)
                        unsup_loss += loss.item()
                        unsup_cnt += 1
                        logging.debug('unsupervised loss:{} m_loss:{} kl_div_loss:{}'.format(loss.item(), m_loss.item(),
                                                                                   kl_div_loss.item()))

            sup_loss /= (sup_cnt + 1e-8)
            unsup_loss /= (unsup_cnt + 1e-8)
        self.eval()
        self.sedst.train()
        return sup_loss, unsup_loss

    def save_model(self, epoch, path=None):
        if not path:
            path = cfg.model_path
        all_state = {'SEDST': self.sedst.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path)
        self.sedst.load_state_dict(all_state['SEDST'])
        self.base_epoch = all_state.get('epoch',0)

    def training_adjust(self, epoch):
        if epoch == cfg.unfrz_attn_epoch:
            self.unfreeze_params()

    def freeze_params(self):
        self.freeze_module(self.sedst.m_decoder.attn_u)
        self.freeze_module(self.sedst.m_decoder.w4)
        self.freeze_module(self.sedst.m_decoder.gate_z)
        self.freeze_module(self.sedst.m_decoder.w5)

    def unfreeze_params(self):
        self.unfreeze_module(self.sedst.m_decoder.attn_u)
        self.unfreeze_module(self.sedst.m_decoder.w4)
        self.unfreeze_module(self.sedst.m_decoder.gate_z)
        self.unfreeze_module(self.sedst.m_decoder.w5)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        initial_arr = self.sedst.u_encoder.embedding.weight.data.cpu().numpy()
        embedding_arr = torch.from_numpy(get_glove_matrix(self.reader.vocab, initial_arr))

        self.sedst.u_encoder.embedding.weight.data.copy_(embedding_arr)
        self.sedst.m_encoder.embedding.weight.data.copy_(embedding_arr)
        self.sedst.m_decoder.emb.weight.data.copy_(embedding_arr)
        self.sedst.qz_decoder.mu.weight.data.copy_(embedding_arr.transpose(1,0))
        self.sedst.pz_decoder.mu.weight.data.copy_(embedding_arr.transpose(1,0))
        if freeze:
            self.freeze_module(self.sedst.u_encoder.embedding)
            self.freeze_module(self.sedst.m_encoder.embedding)
            self.freeze_module(self.sedst.m_decoder.emb)


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

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
    m = Model(args.dataset)
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
