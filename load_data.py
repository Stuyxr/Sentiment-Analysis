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
        model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)
        for word, vector in zip(model.vocab, model.vectors):
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






# get_data(False)




