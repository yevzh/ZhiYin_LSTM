from functools import partial

import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler


class DataSet(Dataset):
    def __init__(self, review, label, method, model):
        self.review = review
        self.label = label
        self.model_name = model
        self.method_name = method
        dataset = list()
        index = 0
        for data in review:
            # split方法将每个单词提取出来作为后续bertToken的输入
            tokens = data.split(' ')
            label_id = label[index]
            index += 1
            dataset.append((tokens, label_id))
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.review)


def my_collate(batch, tokenizer):
    # 对每一个batch的数据进行处理，将文本数据进行Tokenizer化作为后续Bert模型的输入
    tokens, label_ids = map(list, zip(*batch))

    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=320,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    return text_ids, torch.tensor(label_ids)


def load_source(data_file):
    source = (data_file.replace('DataSet', 'Input')).replace('csv', 'xml')
    coms = []
    with open(source, encoding='utf8') as f:
        inf = f.readline()[1:-1]
        while inf:
            if '>' in inf:
                inf = inf[inf.find('>')+1:]
            if '<' in inf:
                inf = inf[:-inf[::-1].find('<')-1]
            if inf:
                coms.append(inf)
            inf = f.readline()
    return coms


def data_loader(tokenizer, batch_size, model_name, method_name, workers, data_file, ratio=0.05, seq=False):
    data = pd.read_csv(data_file, sep=None, header=0, encoding='utf-8', engine='python')
    if seq:
        sample_data = data
    else:
        sample_data = data.sample(frac=ratio, axis=0)
    labels = list(sample_data['labels'])
    comments = list(sample_data['comments'])
    data_set = DataSet(comments, labels, method_name, model_name)
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    if seq:
        seq_sampler = SequentialSampler(data_set)
        loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                            collate_fn=collate_fn, pin_memory=True, sampler=seq_sampler)
        if ('sample' in data_file) or ('input' in data_file):
            comments = load_source(data_file)
    else:
        loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                            collate_fn=collate_fn, pin_memory=True)
    return loader, comments


def load_dataset(tokenizer, train_batch_size, test_batch_size, model_name, method_name, workers, data_file):
    data = pd.read_csv(data_file, sep=None, header=0, encoding='utf-8', engine='python')
    sample_data = data.sample(frac=1.0, axis=0)
    labels = list(sample_data['labels'])
    reviews = list(sample_data['comments'])

    train_reviews, test_reviews, train_labels, test_labels = train_test_split(reviews, labels, train_size=0.9)
    # construct dataset
    train_set = DataSet(train_reviews, train_labels, method_name, model_name)
    test_set = DataSet(test_reviews, test_labels, method_name, model_name)  # DataLoader

    collate_fn = partial(my_collate, tokenizer=tokenizer)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                             collate_fn=collate_fn, pin_memory=True)
    return train_loader, test_loader