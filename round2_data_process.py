# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import os, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import config
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# 保证每次划分数据一致
np.random.seed(41)

def sf_age(x) :
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


def load_and_clean_label(path):
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

def split_data(file2idx, val_ratio=0.1):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    data = set(os.listdir(config.train_dir))
    val = set()
    idx2file = [[] for _ in range(config.num_classes)]
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[idx].append(file)
    for item in idx2file:
        num = int(len(item) * val_ratio)
        val = val.union(item[:num])
    train = data.difference(val)
    return list(train), list(val)

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

def split_all_data(file2idx, val_ratio=0.1):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    data = set(os.listdir(config.round2_train_dir))
    val = set()
    idx2file = [[] for _ in range(config.num_classes)]
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

def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * config.num_classes
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

def split_data_cv(file2idx,kfold=5):
    X_train_cv = []
    X_val_cv   = []
    X = []
    y = np.zeros([len(file2idx),config.num_classes])
    for i,(file, list_idx) in enumerate(file2idx.items()):
        X.append(file)
        y[i,file2idx[file]] = 1
        
    X = np.array(X)

    mskf = MultilabelStratifiedKFold(n_splits=kfold, random_state=42)
    for train_index, test_index in mskf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_val = X[train_index], X[test_index]
        X_train_cv.append(list(X_train))
        X_val_cv.append(list(X_val))
        #y_train, y_test = y[train_index], y[test_index]
    return X_train_cv,X_val_cv

# def train(name2idx, idx2name):
#     file2idx = file2index(config.train_label, name2idx)
#     train, val = split_data(file2idx)
#     wc=count_labels(train,file2idx)
#     print(wc)
#     print(len(train))
#     print(len(val))
#     dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
#     torch.save(dd, config.train_data)

# def train_all_data(name2idx, idx2name):
#     file2idx = file2index(config.train_label, name2idx)
#     train, val = split_all_data(file2idx)
#     wc=count_labels(train,file2idx)
#     print(wc)
#     print(len(train))
#     print(len(val))
#     dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
#     torch.save(dd, config.train_all_data)

def train_cv_data(name2idx, idx2name,labels):

    file2idx = dict()
    for i in range(labels.shape[0]):
        idx = str(labels.iloc[i].id) + ".txt"
        file2idx[idx] = [name2idx[name] for name in labels.iloc[i].arrythmia] 
    print(len(file2idx))
    train_cv, val_cv = split_data_cv(file2idx,config.kfold)
    for i in range(config.kfold):
        wc=count_labels(train_cv[i],file2idx)
        print(len(train_cv[i]))
        print(len(val_cv[i]))
        print(wc)

        wc1=count_labels(val_cv[i],file2idx)
        print(wc1)
        print("**********************************************************************")
        dd = {'train': train_cv[i], 'val': val_cv[i], "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
        torch.save(dd, config.round2_train_data_cv.format(i))

if __name__ == '__main__':
    # arrythmias = get_arrythmias(config.arrythmia)
    # name2idx,idx2name = get_dict(arrythmias)
    labels = load_and_clean_label(config.round2_train_label)

    name2idx = name2index(config.round2_arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    train_cv_data(name2idx, idx2name,labels)

