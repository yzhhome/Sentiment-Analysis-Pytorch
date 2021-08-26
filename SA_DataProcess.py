from io import open
import torch
import re
import numpy as np
import gensim
from torch.utils.data import Dataset
from SA_Config import Config


class Data_set(Dataset):
    def __init__(self, Data, Label) -> None:
        super().__init__()
        self.Data = Data
        if Label is not None:
            self.Label = Label

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 根据索引返回数据
    def __getitem__(self, index):
        if self.Label is not None:
            data = torch.from_numpy(self.Data[index])
            label = torch.from_numpy(np.array(self.Label[index]))
            return data, label
        else:
            data = torch.from_numpy(self.Data[index])
            return data

# 停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open(
        Config.stop_words_path, encoding='utf-8').readlines()]
    return stopwords

# 构建单词到id的映射文件
def build_word2id(save_to_path):
    """
    :param save_to_path: word2id保存的文件路径
    :return None
    """
    stopwords = stopwordslist()
    word2id = {'_PAD_': 0}
    path = [Config.train_path, Config.val_path]
    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                out_list = []
                sp = line.strip().split()

                # 第0个词为标签
                for word in sp[1:]:
                    if word not in stopwords:
                        rt = re.findall('[a-zA-Z]+', word)
                        if word != '\t':
                            # 忽略英文词
                            if len(rt) == 1:
                                continue
                            else:
                                out_list.append(word)

                # word : id 形式添加到word2id字典中
                for word in out_list:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    # 写入到文件中
    with open(save_to_path, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w + '\t')
            f.write(str(word2id[w]))
            f.write('\n')

# 用word2vec预训练模型构建词向量
def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)

    # word_vecs 矩阵初始化为 词典大小 * 词向量维度
    word2vec = np.array(np.random.uniform(-1, 1, [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            # 词对应的id转换成词向量
            word2vec[word2id[word]] = model[word]
        except:
            pass

    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word2vec:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word2vec

# 将句子中的词转换为对应的id数组和标签值
def text_to_array(word2id, seq_lenth, path):

    label_array = []
    i = 0
    sa = []

    # 句子数量
    sa_count = len(open(path, encoding='utf-8').readlines())

    with open(path, encoding='utf-8') as f:
        # 句子id矩阵每个元素初始为0
        sentences_array = np.zeros(shape=(sa_count, seq_lenth))
        for line in f.readlines():
            data = line.strip().split()
            words = data[1:]
            ids = [word2id.get(word, 0) for word in words]
            ids_array = np.array(ids).reshape(1, -1)

            # 比序列长度短的在前面补0
            # sentences_array 后面ids_array长度的值替换为ids_array中的值
            # sentences_array 前面的值还是原来初始的0，相当于在前面补0
            if np.size(ids_array, 1) < seq_lenth:
                sentences_array[i, seq_lenth - np.size(ids_array, 1):] = ids_array[0, :]
            # 比序列长度长的直接截断，只取序列长度
            else:
                sentences_array[i, 0:seq_lenth] = ids_array[0, 0:seq_lenth]

            i = i + 1
            # 取标签值
            label_array.append(int(data[0]))

    return np.array(sentences_array), label_array

# 将句子中的词转换为对应的id数组
def text_to_array_no_label(word2id, seq_lenth, path):

    i = 0
    sa = []

    # 句子数量
    sa_count = len(open(path, encoding='utf-8').readlines())

    with open(path, encoding='utf-8') as f:
        # 句子id矩阵每个元素初始为0
        sentences_array = np.zeros(shape=(sa_count, seq_lenth))
        for line in f.readlines():
            data = line.strip().split()
            words = data[1:]
            ids = [word2id.get(word, 0) for word in words]
            ids_array = np.array(ids).reshape(1, -1)

            # 比序列长度短的在前面补0
            # sentences_array 后面ids_array长度的值替换为ids_array中的值
            # sentences_array 前面的值还是原来初始的0，相当于在前面补0
            if np.size(ids_array, 1) < seq_lenth:
                sentences_array[i, seq_lenth - np.size(ids_array, 1):] = ids_array[0, :]
            # 比序列长度长的直接截断，只取序列长度
            else:
                sentences_array[i, 0:seq_lenth] = ids_array[0, 0:seq_lenth]

            i = i + 1

    return np.array(sentences_array)

# 准备训练数据，验证数据，测试数据
def prepare_data(word2id, train_path, val_path, test_path, seq_lenth):

    train_array, train_lable = text_to_array(word2id, seq_lenth, train_path)
    val_array, val_lable = text_to_array(word2id, seq_lenth, val_path)
    test_array, test_lable = text_to_array(word2id, seq_lenth, test_path)

    train_lable = np.array(train_lable).T
    val_lable = np.array(val_lable).T
    test_lable = np.array(test_lable).T

    return train_array, train_lable, val_array, val_lable, test_array, test_lable

# 准备训练数据，验证数据
def prepare_train_data(word2id, train_path, val_path, seq_lenth):

    train_array, train_lable = text_to_array(word2id, seq_lenth, train_path)
    val_array, val_lable = text_to_array(word2id, seq_lenth, val_path)

    train_lable = np.array(train_lable).T
    val_lable = np.array(val_lable).T

    return train_array, train_lable, val_array, val_lable

# 准备训练数据，验证数据
def prepare_test_data(word2id, test_path, seq_lenth):

    test_array, test_lable = text_to_array(word2id, seq_lenth, test_path)
    test_lable = np.array(test_lable).T
    return test_array, test_lable


if __name__ == '__main__':
    # 构建训练集和验证集的词到id表示
    build_word2id(save_to_path=Config.word2id_path)
    
    splist = []
    word2id = {}

    # 转换为字典 word : id
    with open(Config.word2id_path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            splist.append(sp)
        word2id = dict(splist)

    for key in word2id:
        word2id[key] = int(word2id[key])

    # 转换为字典 id : word
    id2word = {}
    for key, value in word2id.items():
        id2word[value] = key

    # 构建所有词的词向量
    word2vec = build_word2vec(Config.pre_word2vec_path,
                              word2id, Config.corpus_word2vec_path)

    # 转换句子id表示和标签
    train_array, train_lable, \
    val_array, val_lable, \
    test_array, test_label = prepare_data(word2id,
                                        Config.train_path,
                                        Config.val_path,
                                        Config.test_path, 
                                        Config.max_sen_len)

    # 保存训练数据，验证数据，测试数据的id表示形式
    np.savetxt(Config.train_data_path, train_array, fmt='%d')
    np.savetxt(Config.val_data_path, val_array, fmt='%d')
    np.savetxt(Config.test_data_path, test_array, fmt='%d')