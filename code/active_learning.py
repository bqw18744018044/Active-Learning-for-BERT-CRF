# encoding: utf-8
"""
@author: bqw
@time: 2020/5/13 9:36
@file: active_learning.py
@desc: 
"""
import logging
import shutil
from model import Model
from enums import STRATEGY
from active_utils import DataPool, ActiveStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def start_active_learning(train, dev, model_config, active_config, network):
    model = Model(model_config, network)  # 实例化模型(具体来说是BERT-CRF)
    selected_texts = None
    selected_labels = None

    logger.info("The active strategy is {}".format(active_config.select_strategy))
    # STRATEGY.RAND表示随机样本选择策略
    if active_config.select_strategy == STRATEGY.RAND:
        dataPool = DataPool(train['texts'], train['labels'], 0)
        unselected_texts, unselected_labels = dataPool.get_unselected()  # 未被选择的样本及标签
        # 使用随机样本选择策略挑选样本
        tobe_selected_idx = ActiveStrategy.random_sampling(unselected_texts, active_config.total_num)
        dataPool.update(tobe_selected_idx)
        selected_texts, selected_labels = dataPool.get_selected()
        logger.info("The final size of selected data is {}".format(len(selected_texts)))
        model.config.epochs = active_config.total_epochs
        model.config.update()
        if len(selected_texts) == 0:
            model.eval(dev['texts'], dev['texts'])
        else:
            model.train_and_eval(selected_texts, selected_labels, dev['texts'], dev['labels'])
    else:
        # 实例化DataPool
        dataPool = DataPool(train['texts'], train['labels'], active_config.select_num)
        model.config.epochs = active_config.select_epochs
        model.config.update()
        while (selected_texts is None) or len(selected_texts) < active_config.total_num - 5:
            # 划分出被选择样本与未被选择样本
            selected_texts, selected_labels = dataPool.get_selected()
            unselected_texts, unselected_labels = dataPool.get_unselected()
            logger.info("The size of selected data is {}".format(len(selected_texts)))
            logger.info("The size of unselected data is {}".format(len(unselected_texts)))
            logger.info("Query strategy is {}".format(active_config.select_strategy))
            if len(selected_texts) != 0:
                model.train(selected_texts, selected_labels)  # 使用选择出的样本训练模型
            # 使用不同的样本选择策略对未被选择的样本进行评估，并挑选出合适的样本加入到selected_texts中
            if active_config.select_strategy == STRATEGY.LC:
                scores = model.predict_viterbi_score(unselected_texts)
                tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.lc_sampling(scores, unselected_texts,
                                                                                      active_config.select_num)
            elif active_config.select_strategy == STRATEGY.MNLP:
                scores = model.predict_mnlp_score(unselected_texts)
                tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.mnlp_sampling(scores, unselected_texts,
                                                                                        active_config.select_num)
            elif active_config.select_strategy == STRATEGY.TTE:
                probs = model.predict_probs(unselected_texts)
                tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.tte_sampling(probs, unselected_texts,
                                                                                       active_config.select_num)
            elif active_config.select_strategy == STRATEGY.TE:
                probs = model.predict_probs(unselected_texts)
                tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.te_sampling(probs, unselected_texts,
                                                                                      active_config.select_num)
            # 更新DataPool，也就是将tobe_selected_idxs代表的样本加入到selected_texts，并从unselected_texts中删除
            dataPool.update(tobe_selected_idxs)

        #  使用最终挑选出的样本重新训练整个模型
        shutil.rmtree(model.config.model_dir)
        model_config.epochs = active_config.total_epochs
        model = Model(model_config, network)
        logger.info("The max size of selected data is {}".format(active_config.total_num))
        logger.info("The size of selected data is {}".format(len(selected_texts)))
        logger.info("The size of unselected data is {}".format(len(unselected_texts)))
        model.train_and_eval(selected_texts, selected_labels, dev['texts'], dev['labels'])
