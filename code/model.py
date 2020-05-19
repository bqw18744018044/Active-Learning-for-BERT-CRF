# encoding: utf-8
"""
@author: bqw
@time: 2020/5/11 16:42
@file: model.py
@desc: 模型的训练及评估
"""
import tensorflow as tf
import tf_metrics
import functools
import numpy as np

from network import BertCrf
from bert import tokenization
from bert import modeling
from bert import optimization
from configs import ModelConfig
from utils.DataUtils import NERDUtil as ndu


class InputExample(object):
    def __init__(self, guid, text, label=None, segment_id=None):
        self.guid = guid  # ID
        self.text = text  # 文本
        self.label = label  # 标签
        self.segment_id = segment_id  # 段落ID(不使用)


class Model(object):
    def __init__(self, config=None, network=BertCrf):
        if not config:
            self.config = ModelConfig()
            self.config.update()
        else:
            self.config = config
        self.network = network
        # 加载bert参数
        self.bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_file)
        # 实例化estimator(可用于后续的训练、评估等)
        self.estimator = self._build_estimator()
        self.train_examples = None
        self.dev_examples = None
        self.train_features = None
        self.dev_features = None

    def train_and_eval(self, tr_texts, tr_labels, eval_texts, eval_labels, tr_segment_id=None, dev_segment_id=None):
        # 将样本转换为InputExample
        self.train_examples = self._create_examples(tr_texts, tr_labels, tr_segment_id, set_type='train')
        self.eval_examples = self._create_examples(eval_texts, eval_labels, dev_segment_id, set_type='eval')
        # 将InputExample转换为模型可以使用的
        self.train_features = self._convert_examples_to_features(self.train_examples)
        self.eval_features = self._convert_examples_to_features(self.eval_examples)
        # 构建输入函数
        train_inpf = functools.partial(self._input_fn, self.train_features, self.config.train_batch_size, True)
        eval_inpf = functools.partial(self._input_fn, self.eval_features, self.config.eval_batch_size, False)
        num_train_size = len(self.train_examples)
        num_train_steps = int(
            num_train_size / self.config.train_batch_size * self.config.num_train_epochs)
        tf.logging.info("  Num examples = %d", num_train_size)
        tf.logging.info("  Batch size = %d", self.config.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf,
                                            max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=10)
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def eval(self, texts, labels, segment_id=None, batch_size=None):
        if batch_size:
            self.config.eval_batch_size = batch_size
        self.eval_examples = self._create_examples(texts, labels, segment_id, set_type='dev')
        self.eval_features = self._convert_examples_to_features(self.eval_examples)
        eval_inpf = functools.partial(self._input_fn, self.eval_features, self.config.eval_batch_size, False)
        num_eval_size = len(self.eval_examples)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", num_eval_size)
        tf.logging.info("  Batch size = %d", self.config.eval_batch_size)
        self.estimator.evaluate(input_fn=eval_inpf)

    def train(self, texts, labels, segment_id=None, epochs=None, batch_size=None):
        if epochs:
            self.config.num_train_epochs = epochs
        if batch_size:
            self.config.train_batch_size = batch_size
        self.train_examples = self._create_examples(texts, labels, segment_id, set_type='train')
        self.train_features = self._convert_examples_to_features(self.train_examples)
        train_inpf = functools.partial(self._input_fn, self.train_features, self.config.train_batch_size, True)
        num_train_size = len(self.train_examples)
        num_train_steps = int(
            num_train_size / self.config.train_batch_size * self.config.num_train_epochs)
        tf.logging.info("  Num examples = %d", num_train_size)
        tf.logging.info("  Batch size = %d", self.config.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        if self.config.max_seq_length > self.bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.config.max_seq_length, self.bert_config.max_position_embeddings))
        self.estimator.train(input_fn=train_inpf, max_steps=num_train_steps)

    def predict(self, texts):
        segment_ids = list(range(len(texts)))
        features, examples = self._convert_texts_to_features(texts, segment_ids)
        predict_inpf = functools.partial(self._input_fn, features, self.config.predict_batch_size, False)
        preds = self.estimator.predict(input_fn=predict_inpf)
        preds = [pred for pred in preds]
        return preds

    def predict_mnlp_scores(self, texts):
        preds = self.predict(texts)
        scores = [pred['mnlp_score'] for pred in preds]
        return scores

    def predict_viterbi_scores(self, texts):
        preds = self.predict(texts)
        scores = [pred['score'] for pred in preds]
        return scores

    def predict_probs(self, texts):
        preds = self.predict(texts)
        probs = [pred['probs'] for pred in preds]
        return probs

    def _convert_texts_to_features(self, texts, segment_ids=None):
        lines = ndu.convert_to(texts, segment_ids=segment_ids)
        texts = [line[0] for line in lines]
        labels = [line[1] for line in lines]
        segment_id = [line[2] for line in lines]
        examples = self._create_examples(texts, labels, segment_id, set_type='test')
        features = self._convert_examples_to_features(examples)
        return features, examples

    def _input_fn(self, features, batch_size, is_training=False):
        dataset = tf.data.Dataset.from_tensor_slices(features)
        if is_training:
            dataset = dataset.shuffle(self.config.buffer_size).repeat()
        dataset = dataset.batch(batch_size)
        return dataset

    def _create_examples(self, texts, labels, segment_ids=None, set_type='train'):
        """将texts,labels转换为InputExample"""
        examples = []
        assert len(texts) == len(labels)
        for i in range(len(texts)):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(texts[i])
            label = tokenization.convert_to_unicode(labels[i])
            if set_type == "test":
                segment_id = tokenization.convert_to_unicode(segment_ids[i])
                examples.append(InputExample(guid=guid, text=text, label=label, segment_id=segment_id))
            else:
                examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _convert_examples_to_features(self, examples):
        """
        将整个list中的InputExample转换为feature
        """
        features = []
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []
        for (ex_index, example) in enumerate(examples):
            feature = self._convert_single_example(ex_index,
                                                   example,
                                                   self.config.label2id,
                                                   self.config.tokenizer,
                                                   self.config.max_seq_length)
            all_input_ids.append(feature['input_ids'])
            all_input_mask.append(feature['input_mask'])
            all_segment_ids.append(feature['segment_ids'])
            all_label_ids.append(feature['label_ids'])
        input_ids_mat = np.array(all_input_ids)  # 输入举证
        input_mask_mat = np.array(all_input_mask)
        segment_ids_mat = np.array(all_segment_ids)
        label_ids_mat = np.array(all_label_ids)  # 标签矩阵
        features = {'input_ids': input_ids_mat, 'input_mask': input_mask_mat, 'segment_ids': segment_ids_mat,
                    'label_ids': label_ids_mat}
        return features

    def _convert_single_example(self, ex_index, example, label2id, tokenizer, max_seq_length):
        """
        将单个InputExamples转换为feature
        """
        textlist = example.text.split()
        labellist = example.label.split()
        if example.segment_id:
            segment_id = eval(example.segment_id.split()[0])
            segment_ids = [segment_id] * max_seq_length
        else:
            segment_ids = [0] * max_seq_length
        tokens = []
        label_ids = []
        # 经过bert的tokenize处理
        for word, label in zip(textlist, labellist):
            # 通常一个word会转换成一个token，但是有些特殊的字符可能会转换成多个
            # 例如有些日文会转换为多个字符，这样会造成tokens与labels个数不对应，因此引入了"X"标签
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    label_ids.append(label2id[label])
                else:
                    label_ids.append(label2id['X'])

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首标志[CLS]和句尾标志[SEP]
            label_ids = label_ids[0:(max_seq_length - 2)]

        # 加开始标志
        tokens.insert(0, "[CLS]")
        label_ids.insert(0, label2id["[CLS]"])
        # 加结束标志
        tokens.append("[SEP]")
        label_ids.append(label2id["[SEP]"])
        # 将token转换为id
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # 生成input_mask和segment_ids
        input_mask = [1] * len(input_ids)

        # padding
        if len(input_ids) < max_seq_length:
            padding_size = max_seq_length - len(input_ids)
            padding_seq = [0] * padding_size
            input_ids.extend(padding_seq)
            label_ids.extend(padding_seq)
            input_mask.extend(padding_seq)
            tokens.extend(["**NULL**"] * padding_size)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        feature = {'input_ids': input_ids,
                   'input_mask': input_mask,
                   'segment_ids': segment_ids,
                   'label_ids': label_ids}
        return feature

    def _build_estimator(self):
        model_fn = self._model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(self.config.label_list),
            init_checkpoint=self.config.init_checkpoint,
            learning_rate=self.config.learning_rate,
            max_seq_length=self.config.max_seq_length,
            num_train_steps=2000)
        session_config = tf.ConfigProto(log_device_placement=True)
        # session_config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
        run_config = tf.estimator.RunConfig(model_dir=self.config.model_dir,
                                            save_checkpoints_steps=self.config.save_checkpoints_steps).replace(
            session_config=session_config)
        estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
        return estimator

    def _model_fn_builder(self, bert_config, num_labels, init_checkpoint, learning_rate, max_seq_length,
                          num_train_steps=None, num_warmup_steps=None, use_tpu=False,
                          use_one_hot_embeddings=False):
        """
        为estimator创建model_fn
        """

        def model_fn(features, labels, mode, params):
            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
            input_ids = features["input_ids"]  # (batch_size,max_seq_len)
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            net = self.network(bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                          num_labels, max_seq_length, use_one_hot_embeddings)

            tvars = tf.trainable_variables()  # 获取可训练的变量
            if init_checkpoint:
                # 使用init_checkpoint恢复模型参数
                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                           init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            # 输出可训练变量
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(
                    net.loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=net.loss,
                    train_op=train_op)
            elif mode == tf.estimator.ModeKeys.EVAL:
                def metric_fn(label_ids, logits):
                    # 正样本label对应的id
                    positive_pos = [i + 1 for i in range(num_labels - 4)]
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    metrics_dict = {}
                    for entity_type in self.config.entity_types:
                        metrics_dict[entity_type+'_precision'] = tf_metrics.precision(label_ids,
                                                                                      predictions,
                                                                                      num_labels+1,
                                                                                      self.config.entity_map[entity_type],
                                                                                      average="macro")

                        metrics_dict[entity_type + '_recall'] = tf_metrics.recall(label_ids,
                                                                                  predictions,
                                                                                  num_labels + 1,
                                                                                  self.config.entity_map[entity_type],
                                                                                  average="macro")
                        metrics_dict[entity_type + '_f1'] = tf_metrics.f1(label_ids,
                                                                          predictions,
                                                                          num_labels + 1,
                                                                          self.config.entity_map[entity_type],
                                                                          average="macro")
                    all_precision = tf_metrics.precision(label_ids, predictions, num_labels + 1, positive_pos,
                                                     average="macro")
                    all_recall = tf_metrics.recall(label_ids, predictions, num_labels + 1, positive_pos, average="macro")
                    all_f = tf_metrics.f1(label_ids, predictions, num_labels + 1, positive_pos, average="macro")
                    metrics_dict['all_precision'] = all_precision
                    metrics_dict['all_recall'] = all_recall
                    metrics_dict['all_f1'] = all_f
                    return metrics_dict

                eval_metrics = metric_fn(label_ids, net.logits)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=net.loss,
                    eval_metric_ops=eval_metrics)
            else:
                # 进行预测时的输出内容
                pred_dict = {}
                pred_dict['logits'] = net.logits
                pred_dict['predicts'] = net.predicts
                pred_dict['score'] = net.score
                pred_dict['probs'] = net.probs
                pred_dict['best_probs'] = net.best_probs
                pred_dict['mnlp_score'] = net.mnlp_score
                output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_dict)
            return output_spec

        return model_fn
