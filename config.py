# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import os


class Config:
    # for data_process.py
    #root = r'D:\ECG'
    root = r'/tcdata'

    ####round1
    round1_train_dir = os.path.join(root, 'train')
    round1_test_dir = os.path.join(root, 'testA')
    round1_beat_dir = os.path.join('', 'hf_round1_beat_500')
    round1_train_data = os.path.join('', 'train_round1_no_resample.pth')


    round1_train_label = os.path.join(root, 'hf_round1_label.txt')
    round1_test_label = os.path.join(root, 'hefei_round1_ansA_20191008.txt')
    round1_arrythmia = os.path.join(root, 'hf_round1_arrythmia.txt')

    ####round2
    round2_train_dir = os.path.join(root, 'hf_round2_train')
    round2_beat_dir = os.path.join('', 'hf_round2_beat_500')

    round2_test_dir = os.path.join(root, 'hf_round2_testB')
    round2_test_beat_dir = os.path.join('', 'hf_round2_test_beat_500')

    round2_train_data_cv = os.path.join('', 'train_round2_fold{}.pth')

    round2_train_label = os.path.join(root, 'hf_round2_train.txt')
    round2_test_label = os.path.join(root, 'hf_round2_subB.txt')
    round2_arrythmia = os.path.join(root, 'hf_round2_arrythmia.txt')

    #保存模型的文件夹
    ckpt = 'ckpt'
    val_path = "val_cv"
    # for train
    round1_pretrain_mixnet_sm_weight = os.path.join(ckpt,'mixnet_sm_pretrain','transform_best_weight.pth')
    round1_pretrain_mixnet_mm_weight = os.path.join(ckpt,'mixnet_mm_pretrain','transform_best_weight.pth')

    #训练的模型名称
    model_name = "mixnet_mm"  # "mixnet_sm"# 

    model_names = ["mixnet_sm_predict","mixnet_mm_predict"]

    model_ckpts = ["mixnet_sm_cv","mixnet_mm_cv"]
    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [32,64,80]#[24,48,72,84] #[32,64,80]#
    #训练时的batch大小
    batch_size = 32
    #label的类别数
    num_classes = 34#55
    #最大训练多少个epoch
    max_epoch = 50 #100#256
    #目标的采样长度
    target_point_num = 2560#2048

    #保存提交文件的文件夹
    sub_dir = 'submit'
    #初始的学习率
    lr = 1e-3

    pre_lr = 1e-4

    #保存模型当前epoch的权重
    current_w = 'current_weight.pth'
    #保存最佳的权重
    best_w = 'transform_best_weight.pth'

    # 学习率衰减 lr/=lr_decay
    lr_decay = 10

    kfold = 5
    #保存模型当前epoch的权重
    current_w_cv = 'current_weight_fold{}.pth'
    #保存最佳的权重
    best_w_cv = 'best_weight_fold{}.pth'
    #for test
    temp_dir=os.path.join(root,'temp')


config = Config()
