# encoding: utf-8
"""
@author: bqw
@time: 2020/5/13 15:59
@file: run.py
@desc: 
"""
import logging
from enums import STRATEGY
from configs import ModelConfig, ActiveConfig
from network import BertCrf
from active_learning import start_active_learning
from utils.DataUtils import NERDUtil as ndu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run():
    # 读取数据
    train_file = './data/MSRA/train.txt'
    dev_file = './data/MSRA/dev.txt'
    train_data = ndu.read_ner_data(train_file)
    dev_data = ndu.read_ner_data(dev_file)
    tr_texts = [t[0] for t in train_data]
    tr_labels = [t[1] for t in train_data]
    dev_texts = [d[0] for d in dev_data]
    dev_labels = [d[1] for d in dev_data]
    train = {'texts': tr_texts, 'labels': tr_labels}
    dev = {'texts': dev_texts, 'labels': dev_labels}
    # 主动学习参数
    active_config = ActiveConfig(len(train['texts']))
    active_config.select_strategy = STRATEGY.TTE
    active_config.total_percent = 0.5
    active_config.update()
    # 模型参数
    model_config = ModelConfig()
    model_config.model_dir = "../model/MSRA"
    model_config.positive_label = "B-LOC I-LOC B-PER I-PER B-ORG I-ORG"
    model_config.entity_types = ['LOC', 'PER', 'ORG']
    model_config.update()
    start_active_learning(train, dev, model_config, active_config, BertCrf)


if __name__ == '__main__':
    run()