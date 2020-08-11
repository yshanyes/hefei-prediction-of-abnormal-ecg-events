# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch

from config import config

# 保证每次划分数据一致
np.random.seed(41)

def sf_age(x):
    try :
        t = int(x)
    except :
        t = -1
    return t

def sf_gender(x) :
    if x == 'FEMALE' :
        t = 0
    elif x == 'MALE' :
        t = 1    
    else :
        t = -1
    return t

def load_and_clean_sub(path) :
    labels = pd.read_csv(path,header=None,sep='.txt', encoding='utf-8',engine='python')
    res = pd.DataFrame()
    res['id'] = labels[0]

    tmp = []
    for x in labels[1].values:
        try :
            t = x.split('\t')
            tmp.append([t[1],t[2]])
        except :
            tmp.append([np.nan,np.nan])
    tmp = np.array(tmp)
    res['age'] = tmp[:,0]
    res['gender'] = tmp[:,1]
    
    res['age'] = res['age'].apply(sf_age)
    res['gender'] = res['gender'].apply(sf_gender)
    return res


def load_and_clean_label(path) :
    labels = pd.read_csv(path,header=None,sep='.txt', encoding='utf-8',engine='python')
    res = pd.DataFrame()
    res['id'] = labels[0]
    
    tmp = []
    for x in labels[1].values:
        t = x.split('\t')
        tmp.append([t[1],t[2], t[3:]])

    tmp = np.array(tmp)
    res['age'] = tmp[:,0]
    res['gender'] = tmp[:,1]
    res['arrythmia'] = tmp[:,2]
    
    res['age'] = res['age'].apply(sf_age)
    res['gender'] = res['gender'].apply(sf_gender)
    return res

def get_diff_dict(arrythmias):
    str2ids = {}
    id2strs = {}
    for i,a in enumerate(arrythmias):
        str2ids[a] = i+34
        id2strs[i+34] = a
    return str2ids,id2strs


def name2index(path):
    '''
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    '''
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx

def split_data(file2idx, labels_train,labels_val,val_ratio=0.1):
    train = labels_train.id.apply(lambda x:str(x)+".txt").values.tolist()
    val = labels_val.id.apply(lambda x:str(x)+".txt").values.tolist()
    return train, val

# def split_data(file2idx, val_ratio=0.1):
#     '''
#     划分数据集,val需保证每类至少有1个样本
#     :param file2idx:
#     :param val_ratio:验证集占总数据的比例
#     :return:训练集，验证集路径
#     '''
#     data = set(os.listdir(config.train_dir))
#     val = set()
#     idx2file = [[] for _ in range(config.num_classes)]
#     for file, list_idx in file2idx.items():
#         for idx in list_idx:
#             idx2file[idx].append(file)
#     for item in idx2file:
#         # print(len(item), item)
#         num = int(len(item) * val_ratio)
#         val = val.union(item[:num])
#     train = data.difference(val)
#     return list(train), list(val)

def split_all_data(file2idx, val_ratio=0.1,num_classes=55):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    data = set(os.listdir(config.train_dir))
    val = set()
    idx2file = [[] for _ in range(num_classes)]
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[idx].append(file)
    for item in idx2file:
        # print(len(item), item)
        num = int(len(item) * val_ratio)
        val = val.union(item[:num])
    train = data#.difference(val)
    return list(train), list(val)


def file2index(path, name2idx):
    '''
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    file2index = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        labels = [name2idx[name] for name in arr[3:]]
        # print(id, labels)
        file2index[id] = labels
    return file2index

def count_labels(data, file2idx,num_classes=55):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)

# def count_labels(data, file2idx):
#     '''
#     统计每个类别的样本数
#     :param data:
#     :param file2idx:
#     :return:
#     '''
#     cc = [0] * config.num_classes
#     for fp in data:
#         for i in file2idx[fp]:
#             cc[i] += 1
#     return np.array(cc)

def get_arrythmias(arrythmias_path):
    with open(arrythmias_path,"r") as f:
        data = f.readlines()
    arrythmias = [d.strip() for d in data]
    return arrythmias

def get_dict(arrythmias):
    str2ids = {}
    id2strs = {}
    for i,a in enumerate(arrythmias):
        str2ids[a] = i
        id2strs[i] = a
    return str2ids,id2strs

def train(name2idx, idx2name):
    file2idx = file2index(config.train_label, name2idx)
    train, val = split_data(file2idx)
    wc=count_labels(train,file2idx)
    print(wc)
    print(len(train))
    print(len(val))
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
    torch.save(dd, config.train_data)

def train_all_data(name2idx, idx2name):
    file2idx = file2index(config.train_label, name2idx)
    train, val = split_all_data(file2idx)
    wc=count_labels(train,file2idx)
    print(wc)
    print(len(train))
    print(len(val))
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
    torch.save(dd, config.train_all_data)

def train_uncross(name2idx, idx2name,labels_round1):
    
    file2idx = dict()
    for i in range(labels_round1.shape[0]):
        idx = str(labels_round1.iloc[i].id) + ".txt"
        file2idx[idx] = [name2idx[name] for name in labels_round1.iloc[i].arrythmia] 
    print(len(file2idx))
    
    train,val = split_data(file2idx,labels_round1_train,labels_round1_subA)
    # print(len(train))
    # print(len(val))
    
    # wc,train_enhance = count_labels_uncross(train,val,file2idx,idx2name)
    # #train = train + train_enhance
    # print(len(train_enhance))
    # print(len(train))
    # print(wc)
    wc=count_labels(train,file2idx)
    # print(wc)
    # wc1=count_labels(val,file2idx)
    # print(wc1)
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
    torch.save(dd, config.round1_train_data)#os.path.join(path,'train_round1_no_resample.pth')


if __name__ == '__main__':

    labels_round1_subA = load_and_clean_label(config.round1_test_label)#os.path.join(path,'hf_round1_subA_label.txt')
    labels_round1_train= load_and_clean_label(config.round1_train_label)#os.path.join(path,'hf_round1_label.txt')
    labels_round1 = pd.concat([labels_round1_subA,labels_round1_train])

    #arrythmias = get_arrythmias(config.round1_arrythmia)#os.path.join(path,'hf_round1_arrythmia.txt')
    name2idx_round1 = name2index(config.round1_arrythmia)
    name2idx_round2 = name2index(config.round2_arrythmia)#os.path.join(path,'hf_round2_arrythmia.txt')
    
    
    idx2name_round2 = {idx: name for name, idx in name2idx_round2.items()}
    name2idx,idx2name = get_diff_dict(name2idx_round1.keys() - name2idx_round2.keys())#get_dict(arrythmias)
    name2idx.update(name2idx_round2)
    idx2name.update(idx2name_round2)

    train_uncross(name2idx, idx2name,labels_round1)
    # # arrythmias = get_arrythmias(config.arrythmia)
    # # name2idx,idx2name = get_dict(arrythmias)
    # name2idx = name2index(config.arrythmia)
    # idx2name = {idx: name for name, idx in name2idx.items()}
    # train(name2idx, idx2name)
    # # train_all_data(name2idx, idx2name)
