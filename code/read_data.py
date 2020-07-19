import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytransformers import *
import torch.utils.data as Data
import pickle
import os
import sklearn.model_selection
import sklearn.preprocessing
import joblib
from googletrans import Translator

def get_data(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-chinese', train_aug=False):
    """读取数据，拆分数据集， 使用dataloader构建数据集

    Arguments:
        data_path {str} -- dataset路径: 包含train.csv 和 test.csv
        n_labeled_per_class {int} -- 每个类别的有标签数据的数量

    Keyword Arguments:
        unlabeled_per_class {int} -- 每个类别的无标签数据数据量 默认是5000
        max_seq_len {int} -- 最大序列长度 (default: {256})
        model {str} -- 模型名称 (default: {'bert-base-uncased'})
        train_aug {bool} -- 是否在有标签数据集上进行数据增强 (default: {False})

    """
    # 加载bert的tokenizer类
    tokenizer = BertTokenizer.from_pretrained(model)

    def text_filter(sentence: str) -> str:
        """
        过滤掉非汉字和标点符号和非数字
        :param sentence:
        :return:
        """
        line = sentence.replace('\n', '。')
        # 过滤掉非汉字和标点符号和非数字
        linelist = [word for word in line if
                    word >= u'\u4e00' and word <= u'\u9fa5' or word in ['，', '。', '？', '！',
                                                                        '：'] or word.isdigit()]
        return ''.join(linelist)

    #加载训练集和测试集
    def read_files(data_path):
        X = []
        Y = []
        dirname = os.listdir(data_path)
        for dir in dirname:
            # 循环一个label目录下的所有文件
            files = os.listdir(data_path + '/' + dir)
            for file in files:
                text = ''
                with open(data_path + '/' + dir + '/' + file, encoding="utf8", errors='ignore') as f:
                    for line in f:
                        if line != '\n':
                            text += text_filter(line)
                # 如果文本长度小于10个字符，那么就过滤掉
                if len(text) < 10:
                    continue
                X.append(text)
                Y.append(dir)
        return  X, Y

    train_text, train_labels_text = read_files(data_path + '/train')
    test_text, test_labels_text = read_files(data_path + '/eval')
    train_text = np.array(train_text)
    test_text = np.array(test_text)
    label_model = sklearn.preprocessing.LabelEncoder()
    label_model.fit(train_labels_text)
    train_labels = label_model.transform(train_labels_text)
    test_labels = label_model.transform(test_labels_text)
    # label_model.inverse_transform()
    #保存模型
    joblib.dump(label_model, 'model/label_model.joblib')
    # clf = joblib.load('filename.joblib')  # 导入
    #加1的原因是因为，数字从0开始，变成从1开始
    n_labels = max(test_labels) + 1

    # 拆分出有标签集合，无标签集合，开发集
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
        train_labels, n_labeled_per_class, unlabeled_per_class, n_labels)

    #为每个数据集创建dataset类
    train_labeled_dataset = loader_labeled(train_text[train_labeled_idxs], train_labels[train_labeled_idxs], tokenizer, max_seq_len, train_aug)
    train_unlabeled_dataset = loader_unlabeled(train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, train_aug)
    val_dataset = loader_labeled(train_text[val_idxs], train_labels[val_idxs], tokenizer, max_seq_len)
    test_dataset = loader_labeled(test_text, test_labels, tokenizer, max_seq_len)

    print("有标签数据: {}, 无标签数据： {}, 验证集： {}, 测试集 {}".format(len(
        train_labeled_idxs), len(train_unlabeled_idxs), len(val_idxs), len(test_labels)))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, n_labels


def train_val_split(labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=0):
    """拆分原始训练集到有标签训练集，无标签训练集，验证集

    Arguments:
        labels {list} -- 原始训练集的labels列表
        n_labeled_per_class {int} -- 每一类的有标签数据的数据量
        unlabeled_per_class {int} -- 每一类的无标签数据的数据量
        n_labels {int} -- 类别数目

    Keyword Arguments:
        seed {int} -- [random seed of np.shuffle] (default: {0})，随机数种子

    Returns:
        [list] -- 有标签训练集，无标签训练集，验证集的索引
    """
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    #假设训练集有1000个样本， 前20个是训练集，20到900为无标签训练集，剩余为验证集
    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class: 900])
        val_idxs.extend(idxs[900:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class loader_labeled(Dataset):
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False):
        """
        # 有标签数据的loader, trans_dist 是存储翻译后的结果，
        :param dataset_text:  文本array
        :param dataset_label:  对应的标签array
        :param tokenizer:  使用的tokenizer对象
        :param max_seq_len:  最大序列长度
        :param aug:  是否使用数据增强
        """
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len
        self.aug = aug
        self.trans_dist = {}

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        """
        数据增强, 翻译成英文，在翻译回中文
        :param text: 单个文档的文本
        :return:  新的列表，列表里面是生成后的文本
        """
        translator = Translator(service_urls=['translate.google.cn'])
        if text not in self.trans_dist:
            text1 = translator.translate(text, dest='en')
            text2 = translator.translate(text1.text, dest='zh-cn')
            self.trans_dist[text] = text2.text
        return self.trans_dist[text]

    def get_tokenized(self, text):
        """
        对单个文档的文本text做tokenizer
        :param text: 纯文本内容
        :return:
        """
        tokens = self.tokenizer.tokenize(text)
        #若果大于最大长度，那么截断
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        #获取最后的长度
        length = len(tokens)
        # 把token 转换成id， encode_result是转换成id后的数字列表
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        #如果小于最长长度，那么做padding
        padding = [0] * (self.max_seq_len - len(encode_result))
        #padding 0加到末尾
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        """

        :param idx: 迭代是的索引，int
        :return: 如果数据增强： 原始文本encode后的结果（做了padding），数据增强生成的文本encode后结果，各自对应的labels和未padding时的文本长度
        """
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            #对原始文本和生成的文本都进行encode， text_result是encode后的id列表，text_length是文本的实际长度
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]), (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            #如果不做数据增强，只返回原始文本encode结果，对应label，和原始文本的长度
            return (torch.tensor(encode_result), self.labels[idx], length)


class loader_unlabeled(Dataset):
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, aug=None):
        """
         # 无标签数据加载器
        :param dataset_text:
        :param unlabeled_idxs:
        :param tokenizer:
        :param max_seq_len:
        :param aug:
        """
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        """

        :param text:
        :return:
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        if self.aug is not None:
            u, v, ori = self.aug(self.text[idx], self.ids[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_v, length_v = self.get_tokenized(v)
            encode_result_ori, length_ori = self.get_tokenized(ori)
            return ((torch.tensor(encode_result_u), torch.tensor(encode_result_v), torch.tensor(encode_result_ori)), (length_u, length_v, length_ori))
        else:
            text = self.text[idx]
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)
