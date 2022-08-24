# -*- coding: utf-8 -*-

from functools import partial

import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator
from utils import convert_example, evaluate, predict, load_dict


import os

#4 数据格式调整
def format_data(source_filename, target_filename):
    datalist=[]
    with open(source_filename, 'r', encoding='utf-8') as f:
        lines=f.readlines()
    words=''
    labels=''
    flag=0
    for line in lines:
        if line == '\n':
            item=words+'\t'+labels+'\n'
            # print(item)
            datalist.append(item)
            words=''
            labels=''
            flag=0
            continue
        word, label = line.strip('\n').split(' ')
        if flag==1:
            words=words+'\002'+word
            labels=labels+'\002'+label
        else:
            words=words+word
            labels=labels+label
            flag=1
    with open(target_filename, 'w', encoding='utf-8') as f:
        lines=f.writelines(datalist)
    print(f'{source_filename}文件格式转换完毕，保存为{target_filename}')

# format_data('./dataset/dev.conll', './dataset/dev.txt')
# format_data(r'./dataset/train.conll', r'./dataset/train.txt')
#加载自定义数据集
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

# Create dataset, tokenizer and dataloader.
train_ds, dev_ds = load_dataset(datafiles=(
        './dataset/train.txt', './dataset/dev.txt'))

for i in range(5):
    print(train_ds[i])

#label标签表构建
def gernate_dic(source_filename1, source_filename2, target_filename):
    data_list = []

    with open(source_filename1, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if line != '\n':
            dic = line.strip('\n').split(' ')[-1]
            if dic + '\n' not in data_list:
                data_list.append(dic + '\n')

    with open(source_filename2, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if line != '\n':
            dic = line.strip('\n').split(' ')[-1]
            if dic + '\n' not in data_list:
                data_list.append(dic + '\n')

    with open(target_filename, 'w', encoding='utf-8') as f:
        lines = f.writelines(data_list)
# 从dev文件生成dic
#gernate_dic('dataset/train.conll', 'dataset/dev.conll', 'dataset/mytag.dic')
# gernate_dic('dataset/dev.conll', 'dataset/mytag_dev.dic')

#数据处理
label_vocab = load_dict('./dataset/mytag.dic')
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab)

train_ds.map(trans_func)
dev_ds.map(trans_func)
print (train_ds[0])
#数据读入，使用paddle.io.DataLoader接口多线程已补加载数据
ignore_label = -1
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(),  # seq_len
    Pad(axis=0, pad_val=ignore_label)  # labels
): fn(samples)

train_loader = paddle.io.DataLoader(
    dataset=train_ds,
    batch_size=300,
    return_list=True,
    collate_fn=batchify_fn)
dev_loader = paddle.io.DataLoader(
    dataset=dev_ds,
    batch_size=300,
    return_list=True,
    collate_fn=batchify_fn)
#加载预训练模型
# Define the model netword and its loss
model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))

#设置fine-tune优化策略，模型配置
metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())

#模型训练与评估
step = 0
for epoch in range(50):
    for idx, (input_ids, token_type_ids, length, labels) in enumerate(train_loader):
        logits = model(input_ids, token_type_ids)
        loss = paddle.mean(loss_fn(logits, labels))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        step += 1
        print("epoch:%d - step:%d - loss: %f" % (epoch, step, loss))
    evaluate(model, metric, dev_loader)

    paddle.save(model.state_dict(),
                './checkpoint/model_%d.pdparams' % step)

#模型保存
#!mkdir ernie_result
# model.save_pretrained('./ernie_result')
# tokenizer.save_pretrained('./ernie_result')