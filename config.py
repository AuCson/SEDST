import logging
import time
import configparser


class _Config:
    def __init__(self):
        """
        Important hyperparameters

        self.cuda_device = 0  # major gpu device
        self.aux_device = 1  # auxiliary gpu device, used for implicit copynet
        self.base_epoch = 0  # which epoch to resume
        self.base_iter = 0  # which iteration to resume
        self.beam_len_bonus = 0.5  # bonus for sentence length during beam search decoding
        self.max_turn = 4  # only use final x turns for training
        self.trunc_turn = 2  # truncate the gradient propagation every x turns
        self.last_turn_only = False  # if true, generate response for only the final turn. Used for non-tasked-oriented dialogues
        self.u_max_ts = 100  # max length of previous response + user input
        self.max_ts = 40  # max length of response
        self.freq_thres = 0  # mask the most x words in copying mechanism


        self.q_hidden_size # hidden size of the posterior network
        self.hidden_size # hidden size of the prior network
        self.z_length # max length of the state span
        self.spv_proportion # supervision proportion

        """
        self._init_logging_handler()
        self.force_stable = True
        self.cuda_device = 0
        self.aux_device = 1
        self.base_epoch = 0
        self.eos_m_token = 'EOS_M'
        self.base_iter = 0
        self.beam_len_bonus = 0.5
        self.max_turn = 4
        self.trunc_turn = 2
        self.mode = 'unknown'
        self.m = 'SEDST'
        self.prev_z_method = 'none'
        self.last_turn_only = False
        self.u_max_ts = 100
        self.freq_thres = 0

    def init_handler(self, m):
        init_method = {
            'camrest': self._camrest_init,
            'kvret': self._kvret_init,
            'ubuntu': self._ubuntu_init,
            'jd': self._jd_init
        }
        init_method[m]()

    def _camrest_init(self):
        self.q_hidden_size = 50
        self.vocab_size = 832
        self.embedding_size = 50
        self.hidden_size = 50
        self.split = (3, 1, 1)
        self.lr = 0.003
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab09_slot_space.pkl'
        self.data = './data/CamRest676/CamRest676.json'
        self.entity = './data/CamRest676/CamRestOTGY.json'
        self.db = './data/CamRest676/CamRestDB.json'
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.degree_size = 5
        self.z_length = 8
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100
        self.cuda = True
        self.spv_proportion = 0
        self.alpha = 0.1
        self.max_ts = 40
        self.early_stop_count = 3
        self.new_vocab = True
        self.model_path = './models/camrest.pkl'
        self.result_path = './results/camrest.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 5
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False
        self.trunc_turn = 999
        self.max_turn = 10

    def _kvret_init(self):
        self.intent = 'all'
        self.vocab_size = 1400
        self.embedding_size = 50
        self.hidden_size = 50
        self.split = None
        self.lr = 0.003
        self.q_hidden_size = 50
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab10_slot_space.pkl'
        self.train = './data/kvret/kvret_train_public.json'
        self.dev = './data/kvret/kvret_dev_public.json'
        self.test = './data/kvret/kvret_test_public.json'
        self.entity = './data/kvret/kvret_entities.json'
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.degree_size = 2
        self.z_length = 8
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100
        self.cuda = True
        self.spv_proportion = 0
        self.alpha = 0.1
        self.max_ts = 40
        self.early_stop_count = 5
        self.new_vocab = True
        self.model_path = './models/kvret.pkl'
        self.result_path = './results/kvret.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 5
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False
        self.trunc_turn = 999
        self.max_turn = 10

    def _ubuntu_init(self):
        self.beam_len_bonus = 1.5
        self.vocab_size = 20004
        self.embedding_size = 300
        self.hidden_size = 500
        self.q_hidden_size = 300
        self.split = None
        self.lr = 0.0005
        self.lr_decay = 0.5
        self.vocab_path = './data/ubuntu/Dataset.dict.pkl'
        self.train = './data/ubuntu/Training.dialogues.pkl'
        self.dev = './data/ubuntu/Validation.dialogues.pkl'
        self.test = './data/ubuntu/Test.dialogues.pkl'
        self.glove_path = './data/fasttext/crawl-300d-2M.vec'
        self.batch_size = 32
        self.degree_size = 2
        self.z_length = 5
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100
        self.cuda = True
        self.spv_proportion = 0
        self.alpha = 0.1
        self.max_ts = 60
        self.early_stop_count = 3
        self.new_vocab = True
        self.model_path = './models/ubuntu.pkl'
        self.result_path = './results/ubuntu.csv'
        self.teacher_force = 100
        self.beam_search = True
        self.beam_size = 5
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = True
        self.pretrain = False
        self.trunc_turn = 2

    def _jd_init(self):
        self.beam_len_bonus = 0.5
        self.vocab_size = 20000
        self.embedding_size = 300
        self.hidden_size = 500
        self.q_hidden_size = 300
        self.split = None
        self.lr = 0.0005
        self.lr_decay = 0.5
        self.vocab_path = './vocab/jd20000.pkl'
        self.data = './data/jd/jd.all.seg'
        self.glove_path = './data/fasttext/wiki.zh.vec'
        self.batch_size = 32
        self.degree_size = 2
        self.z_length = 10
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100
        self.cuda = True
        self.spv_proportion = 0
        self.alpha = 0.1
        self.max_ts = 60
        self.early_stop_count = 3
        self.new_vocab = True
        self.model_path = './models/jd.pkl'
        self.result_path = './results/jd.csv'
        self.teacher_force = 100
        self.beam_search = True
        self.beam_size = 5
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = True
        self.pretrain = False
        self.trunc_turn = 2

    def __str__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += '{} : {}\n'.format(k, v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)


global_config = _Config()
