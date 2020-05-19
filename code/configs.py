# encoding: utf-8
"""
@author: bqw
@time: 2020/5/13 15:41
@file: configs.py
@desc: 
"""
import os
import codecs
import pickle
from bert import tokenization
from enums import STRATEGY


class ModelConfig(object):
    def __init__(self):
        self.model_dir = '../model/RM'  # 模型目录
        self.eval_dir = None  # 评估结果目录
        self.label2id_dir = './data'  # label2id.pkl文件目录
        self.bert_path = '../model/chinese_L-12_H-768_A-12'  # 预训练模型BERT的目录
        self.bert_config_file = None  # BERT的config文件路径
        self.init_checkpoint = None  # 初始checkpoint文件路径
        self.vocab_file = None  # 词表文件路径
        self.save_checkpoints_steps = 1000
        self.do_lower_case = True
        self.max_seq_length = 128
        self.num_train_epochs = 5.0
        self.train_batch_size = 16
        self.eval_batch_size = 32
        self.predict_batch_size = 64
        self.buffer_size = 10000
        self.positive_label = "B-LOC I-LOC B-PER I-PER B-ORG I-ORG B-TIME I-TIME"
        # self.positive_label = "B-LOC I-LOC B-PER I-PER B-ORG I-ORG B-BOOK I-BOOK B-COMP I-COMP B-GAME I-GAME B-GOVERN " \
        #                      "I-GOVERN B-MOVIE I-MOVIE B-POS I-POS B-SCENE I-SCENE"
        self.label_list = None
        self.label2id = None
        self.id2label = None
        self.entity_types = ['LOC', 'PER', 'ORG', 'TIME']
        # self.entity_types = ['LOC', 'PER', 'ORG', 'BOOK', 'COMP', 'GAME', 'GOVERN', 'MOVIE', 'POS', 'SCENE']
        self.entity_map = None
        self.learning_rate = 5e-5
        # self.gpu_usage = 0.5
        self.tokenizer = None

    def update(self):
        """由基础的参数生成其他参数"""
        self.eval_dir = os.path.join(self.model_dir, 'eval')
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        self.bert_config_file = os.path.join(self.bert_path, 'bert_config.json')
        self.init_checkpoint = os.path.join(self.bert_path, 'bert_model.ckpt')
        self.vocab_file = os.path.join(self.bert_path, 'vocab.txt')
        self.label_list = self.positive_label.split() + ["O", "X", "[CLS]", "[SEP]"]
        if os.path.exists(os.path.join(self.label2id_dir, 'label2id.pkl')):
            with codecs.open(os.path.join(self.label2id_dir, 'label2id.pkl'), 'rb') as rf:
                self.label2id = pickle.load(rf)
        else:
            self.label2id = {}
            for (i, label) in enumerate(self.label_list, 1):
                self.label2id[label] = i
            with codecs.open(os.path.join(self.label2id_dir, 'label2id.pkl'), 'wb') as f:
                pickle.dump(self.label2id, f)
        self.id2label = {value: key for key, value in self.label2id.items()}
        # entity_map = {'LOC':['B-LOC', 'I-LOC'],...}
        self.entity_map = {entity_type: [self.label2id['B-' + entity_type], self.label2id['I-' + entity_type]]
                           for entity_type in self.entity_types}
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file,
                                                    do_lower_case=self.do_lower_case)


class ActiveConfig(object):
    def __init__(self, pool_size, total_percent=0.5, select_percent=0.25, select_epochs=10, total_epochs=20):
        self.pool_size = pool_size
        self.total_percent = total_percent
        self.select_percent = select_percent
        self.total_num = None
        self.select_num = None
        self.select_strategy = STRATEGY.MNLP
        self.select_epochs = select_epochs  # 主动学习时，每次微调时的epoch
        self.total_epochs = total_epochs  # 重新使用完整的数据训练模型的epochs
        self.update()

    def update(self):
        """
        When you reset any parameters, call this method to update the relevant parameters.
        """
        self.total_num = int(self.pool_size*self.total_percent)
        self.select_num = int(self.total_num*self.select_percent)