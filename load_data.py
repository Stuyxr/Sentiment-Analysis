import jieba
import numpy as np
import torch
from gensim.models import KeyedVectors

pos_data_path = './train_data/sample.positive.txt'
neg_data_path = './train_data/sample.negative.txt'
test_data_path = './test_data/test.txt'
cut_data_path = './cut.txt'
vec_data_path = './vec.bin'
train_data_path = './train_data.pt'
label_data_path = './label_data.pt'
test_path = './test_data.pt'
dic = {}


def read_data(path):
    ret = []
    with open(path, 'r', encoding='UTF-8') as f:
        new_line_comments = str()
        for line in f:
            line = line.strip()
            if '<review' in line:
                new_line_comments = str()
                continue
            if len(line) == 0:
                continue
            if '</review' in line:
                ret.append(new_line_comments)
            new_line_comments += line
    return ret

def get_vec(sentence):
    i = 0
    vec = []
    zeros = [0 for i in range(300)]
    for word in sentence.split(' '):
        if i >= 100:
            break
        if word in dic.keys():
            vec.append(dic[word].tolist())
            i += 1
    while i < 100:
        vec.insert(0, zeros)
        i += 1
    return vec

def get_test_data():
    test_raw_data = read_data(test_data_path)
    test_cut_data = []
    for sentence in test_raw_data:
        test_cut_data.append(' '.join(list(jieba.cut(sentence))))
    test_vec_data = []
    for sentence in test_cut_data:
        test_vec_data.append(get_vec(sentence))
    test_vec_data = torch.Tensor(np.array(test_vec_data))
    torch.save(test_vec_data, test_path)
    return test_vec_data

def get_data(load_from_file=True):
    if not load_from_file:
        raw_data = read_data(pos_data_path) + read_data(neg_data_path)
        cut_data = []
        with open(cut_data_path, 'w', encoding='UTF-8') as f:
            f.write('')
        with open(cut_data_path, 'a', encoding='UTF-8') as f:
            for sentence in raw_data:
                cut_data.append(' '.join(list(jieba.cut(sentence))))
            for cut_sentence in cut_data:
                f.write(cut_sentence)
        model = KeyedVectors.load_word2vec_format('sgns.weibo.bigram', binary=False)
        for word, vector in zip(model.key_to_index, model.vectors):
            dic[word] = vector
        vec_data = []
        for sentence in cut_data:
            vec = get_vec(sentence)
            vec_data.append(vec)
        vec_data = torch.Tensor(np.array(vec_data))
        label = torch.Tensor(np.array([1 for i in range(5000)] + [0 for i in range(5000)]))
        vec_data, label = change_order(vec_data, label)
        torch.save(vec_data, train_data_path)
        torch.save(label, label_data_path)
        test_data = get_test_data()
    vec_data = torch.load(train_data_path)
    label = torch.load(label_data_path)
    test_data = torch.load(test_path)
    return vec_data.permute(1, 0, 2), label, test_data.permute(1, 0, 2)

def change_order(set, target):
    permutation = np.random.permutation(target.shape[0])
    return set[permutation, :, :], target[permutation]






get_data(False)

'''

这段代码的主要目的是从正面和负面评论文件中读取数据、进行预处理，并将预处理后的数据保存为 PyTorch 张量以供稍后进行情感分析。以下是代码的详细步骤：

导入所需的库，如 jieba（用于中文分词）、numpy、torch 和 gensim.models.KeyedVectors。

定义文件路径，如正面和负面训练数据、测试数据、分词数据和词向量等。

定义 read_data(path) 函数，用于从给定路径的文件中读取原始评论数据，并返回评论列表。

定义 get_vec(sentence) 函数，用于将分词后的句子转换为词向量序列。对于每个句子，它会选择前 100 个词（如果有的话）并将其转换为词向量。如果句子中的词数少于 100 个，它会在序列前补零以达到 100 个词的长度。

定义 get_test_data() 函数，用于读取、分词和转换测试数据，并将其保存为 PyTorch 张量。

定义 get_data(load_from_file=True) 函数，用于获取训练和测试数据。这个函数有两个主要部分：

如果 load_from_file 为 False（表示从头开始处理数据），它会执行以下操作：
从正面和负面训练数据文件中读取原始评论数据。
对评论数据进行中文分词，并将分词结果保存到文件中。
从预训练的词向量模型（如 sgns.weibo.bigram）中加载词向量。
将分词后的句子转换为词向量序列，并将结果保存为 PyTorch 张量。
为训练数据创建标签张量（正面评论为 1，负面评论为 0）。
随机打乱训练数据和标签的顺序。
将处理后的训练数据和标签张量保存到文件中。
调用 get_test_data() 函数获取测试数据。
如果 load_from_file 为 True（表示从文件中加载已处理的数据），它会直接从文件中加载训练数据、标签和测试数据张量。
定义 change_order(set, target) 函数，用于打乱给定数据集和标签的顺序。

在最后一行，调用 get_data(False) 函数以从头开始处理数据。您可以根据需要将其更改为 get_data(True)，以从文件中加载已处理的数据。

总之，这段代码的主要功能是读取原始评论数据，将其分词并转换为词向量
'''
