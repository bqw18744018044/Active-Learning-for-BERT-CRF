import pandas as pd
import numpy as np
import codecs

class NERDUtil(object):
    """
    命名实体数据集处理工具
    """

    def __init__(self):
        pass

    @classmethod
    def read_ner_data(cls, path, encoding='utf-8', model='bert'):
        """
        读取命名实体识别数据集，并转换为中间格式
        中间格式示例：
        [['海 钓 比 赛 地 点 在 厦 门 与 金 门 之 间 的 海 域 。',
        'O O O O O O O O O O O O O O O O O O'],
        ['这 座 依 山 傍 水 的 博 物 馆 由 国 内 一 流 的 设 计 师 主 持 设 计 ， 整 个 建 筑 群 精 美 而 恢 宏 。',
        'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O']]

        Parameters：
        -----------
        path: 文件路径
        encoding: 文件编码方式
        model: 'bert' or 'ernie'。当model为'bert'时，对应的数据格式为竖列(可以有多列);当model为'ernie'，对应的格式是横列

        Returns：
        -----------
        list，每个元素对应一条文本的相关信息
        """
        if model == 'bert':
            with codecs.open(path, 'r', encoding) as file:
                col_num = len(file.readline().split())
                result = []
                file.seek(0)
                text_cols = [[] for i in range(col_num)]  # 存储一条文本的各列信息
                for line in file:
                    line = line.strip()
                    if len(line) == 0:  # 空白行，表示一条文本的结束
                        text = []
                        for col in text_cols:
                            text.append(' '.join([c for c in col if len(c) > 0]))
                        result.append(text)
                        text_cols = [[] for i in range(col_num)]
                        continue
                    for i, token in enumerate(line.split()):
                        text_cols[i].append(token)
        if model == 'ernie':
            with codecs.open(path, 'r', encoding) as file:
                file.readline()  # 过滤掉第一行
                result = []
                for line in file:
                    text = []
                    cols = line.strip().split()
                    for col in cols:
                        text.append(" ".join(col.split(u"")))
                    result.append(text)
        return result

    @classmethod
    def save_ner_data(cls, data, path, word_sep=' ', line_sep='\r\n', line_break='\r\n', title=['text_a', 'label'],
                      encoding='utf-8', model='bert'):
        """
        将指定格式的ner数据保存到文件中

        Parameters：
        -----------
        data:包含文本和对应标签的list，格式同read_ner_data的输出相同；
        path：数据保存路径；
        word_sep：字(词)与其标签之间的分隔符；
        line_sep：在model为'ernie'时，对应文本和标签的分隔符；
        blank_line：不同文本间的分隔符；
        title：在modelWie'ernie'时，将该title加入到文件中
        encoding：
        model：'bert'、'ernie'
        """
        if not data or len(data) == 0:
            raise Exception('数据不存在.')
        save_list = []
        if model == 'bert':
            for line in data:
                line = pd.DataFrame({str(i): l.split() for i, l in enumerate(line)})
                line = line.T
                for idx in line:
                    save_list.append(word_sep.join(line[idx].tolist()) + line_sep)
                save_list.append(line_break)
        if model == 'ernie':
            word_sep = u""
            line_sep = '\t'
            save_list.append("\t".join(title) + line_break)
            for line in data:
                row = []
                for col in line:
                    new_col = word_sep.join(col.split())
                    row.append(new_col)
                save_list.append(line_sep.join(row) + line_break)
        with codecs.open(path, 'w', encoding=encoding) as file:
            for l in save_list:
                file.write(l)

    @classmethod
    def train_test_split_ner(cls, data, shuffle=True, test_size=0.3, seed=0):
        """
        划分NER数据的训练和测试集

        Parameters：
        -----------
        data:包含文本和对应标签的list，格式同read_ner_data的输出相同；
        shuffle:是否对数据进行混洗；
        test_size:测试集所占比例；
        seed:随机数种子；

        Returns：
        --------
        train：训练集，格式同data
        test：测试集，格式同data
        """
        data = data.copy()
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(data)
        split_idx = int(len(data) * test_size)
        test = data[:split_idx]
        train = data[split_idx:]
        return train, test

    @classmethod
    def convert_to(cls, sentences, segment_ids=None, default_label='O'):
        """
        为给定的文本添加上默认的标签和分段id（不是必须）,主要用于预测时使用；

        Parameters：
        -----------
        sentences：list，包含文本；
        segment_dis：list，包含段落id，即split_text的返回值；
        default_label：默认的标签；

        Returns：
        -----------
        result：包含文本、标签和段落id
        """
        result = []
        if len(sentences) > 0:
            if segment_ids and len(sentences) != len(segment_ids):
                raise Exception('sentences与segment_ids的长度不一致！')
            for i in range(len(sentences)):
                tmp = []
                sentence = sentences[i]
                sen_len = len(sentence)
                new_sentence = " ".join(list(sentence))
                new_label = " ".join([default_label for _ in range(sen_len)])
                tmp.append(new_sentence)
                tmp.append(new_label)
                if segment_ids:
                    new_segment_id = " ".join([str(segment_ids[i]) for _ in range(sen_len)])
                    tmp.append(new_segment_id)
                result.append(tmp)
        return result