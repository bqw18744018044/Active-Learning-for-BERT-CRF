# encoding: utf-8
"""
@author: bqw
@time: 2020/5/13 9:35
@file: active_utils.py
@desc: 
"""

import random
import numpy as np


class DataPool(object):
    """
    用于管理样本，维护已选择样本与未选择样本
    """
    def __init__(self, texts, labels, init_num):
        self.text_pool = np.array(texts)
        self.label_pool = np.array(labels)
        assert len(texts) == len(labels)
        self.pool_size = len(texts)
        self.selected_texts = None
        self.selected_labels = None
        self.unselected_texts = None
        self.unselected_labels = None
        self.selected_idx = sorted(set(random.sample(list(range(self.pool_size)), init_num)))
        self.unselected_idx = sorted(set(range(self.pool_size)) - set(self.selected_idx))
        self.update_pool()

    def update_pool(self):
        self.selected_texts = self.text_pool[self.selected_idx]
        self.selected_labels = self.label_pool[self.selected_idx]
        self.unselected_texts = self.text_pool[self.unselected_idx]
        self.unselected_labels = self.label_pool[self.unselected_idx]

    def update_idx(self, new_selected_idx):
        new_selected_idx = set(new_selected_idx)
        # 将new_selected_idx样本加入到selected_idx
        self.selected_idx = sorted(set(self.selected_idx) | new_selected_idx)
        # 将new_selected_idx从unselected_idx中删除
        self.unselected_idx = sorted(set(self.unselected_idx) - new_selected_idx)

    def translate_select_idx(self, source_idx):
        target_idx = [self.unselected_idx[idx] for idx in source_idx]
        return target_idx

    def update(self, unselected_idx):
        unselected_idx = self.translate_select_idx(unselected_idx)
        self.update_idx(unselected_idx)
        self.update_pool()

    def get_selected(self):
        return self.selected_texts, self.selected_labels

    def get_unselected(self):
        return self.unselected_texts, self.unselected_labels


class ActiveStrategy(object):
    def __init__(self):
        pass

    @classmethod
    def random_sampling(cls, texts, num):
        idxs = list(range(len(texts)))
        if num > len(texts):
            return idxs
        return random.sample(idxs, num)

    @classmethod
    def lc_sampling(cls, viterbi_scores, texts, select_num):
        """
        Least Confidence
        """
        select_num = select_num if len(texts) >= select_num else len(texts)
        scores = np.array(viterbi_scores)
        tobe_selected_idxs = np.argsort(scores)[:select_num]
        tobe_selected_scores = scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    @classmethod
    def mnlp_sampling(cls, mnlp_scores, texts, select_num):
        select_num = select_num if len(texts) >= select_num else len(texts)
        scores = np.array(mnlp_scores)
        tobe_selected_idxs = np.argsort(scores)[:select_num]
        tobe_selected_scores = scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    @classmethod
    def total_token_entropy(cls, prob):
        epsilon = 1e-9
        prob += epsilon
        tte = np.einsum('ij->', -np.log(prob) * prob)
        return tte

    @classmethod
    def tte_sampling(cls, probs, texts, select_num):
        """
        Total token entropy sampling.
        """
        select_num = select_num if len(texts) >= select_num else len(texts)
        tte_scores = np.array([cls.total_token_entropy(prob[:len(text), :])
                               for prob, text in zip(probs, texts)])
        tobe_selected_idxs = np.argsort(tte_scores)[-select_num:]
        tobe_selected_scores = tte_scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    @classmethod
    def te_sampling(cls, probs, texts, select_num):
        select_num = select_num if len(texts) >= select_num else len(texts)
        te_scores = np.array([cls.total_token_entropy(prob[:len(text), :])/len(text)
                              for prob, text in zip(probs, texts)])
        tobe_selected_idxs = np.argsort(te_scores)[-select_num:]
        tobe_selected_scores = te_scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores