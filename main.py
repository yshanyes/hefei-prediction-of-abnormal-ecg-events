# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ECGDataset
from config import config
from tqdm import tqdm
import radam
from torch.autograd import Variable
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)

# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt_cv(state, is_best, model_save_dir,fold):
    current_w = os.path.join(model_save_dir, config.current_w_cv.format(fold))
    best_w = os.path.join(model_save_dir, config.best_w_cv.format(fold))
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
    return loss_meter / it_count, f1_meter / it_count

def train_beat_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, beat, target in train_dataloader:
        inputs = inputs.to(device)
        beat = beat.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs,beat)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
    return loss_meter / it_count, f1_meter / it_count

def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1 = utils.calc_f1(target, output, threshold)
            f1_meter += f1
    return loss_meter / it_count, f1_meter / it_count

def val_beat_epoch(model, criterion, val_dataloader, threshold=0.5,model_name=None,save=False,fold=0,train=False):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    y_true = []
    y_prob = []

    with torch.no_grad():
        for inputs, beat, target in val_dataloader:
            inputs = inputs.to(device)
            beat = beat.to(device)
            target = target.to(device)
            output = model(inputs,beat)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)

            if save:
                y_t, y_p, f1 = utils.re_calc_f1(target, output, threshold)
                y_true.append(y_t)
                y_prob.append(y_p)
            else:
                f1 = utils.calc_f1(target, output, threshold)

            f1_meter += f1

    if save:
        if train:
            output_proba_file = '%s_fold%s_train_proba.npy' % (model_name,fold)
            output_target_file = '%s_fold%s_train_target.npy' % (model_name,fold)
        else:
            output_proba_file = '%s_fold%s_test_proba.npy' % (model_name,fold)
            output_target_file = '%s_fold%s_test_target.npy' % (model_name,fold)

        np.save(os.path.join(config.val_path,output_target_file),np.vstack(y_true))# train_y_true.npy val_y_true
        np.save(os.path.join(config.val_path,output_proba_file),np.vstack(y_prob))# train_y_prob.npy val_y_prob

    return loss_meter / it_count, f1_meter / it_count

#python round1_beat_process.py
#python round1_data_process.py
#python main.py train --model_name=mixnet_sm_pretrain
def train(args):
    # model
    model = getattr(models, args.model_name)()
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])

    # ignored_params = list(map(id, model.fc.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params,
    #                      model.parameters())

    # optimizer = torch.optim.SGD([
    #             {'params': base_params},
    #             {'params': model.fc.parameters(), 'lr': 1e-2}
    #             ], lr=1e-3, momentum=0.9)

    # for param in model.parameters():
    #     param.requires_grad = False

    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, config.num_classes)
    model = model.to(device)

    # data
    train_dataset = ECGDataset(data=config.round1_train_data, data_path=config.round1_train_dir, beat_path=config.round1_beat_dir, train=True,num_classes=55)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)

    val_dataset = ECGDataset(data=config.round1_train_data, data_path=config.round1_test_dir, beat_path=config.round1_beat_dir, train=False,num_classes=55)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)

    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = radam.RAdam(model.parameters(), lr=config.lr) # model.classifier.parameters() optim.Adam(model.parameters(), lr=config.lr)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = utils.WeightedMultilabel(w) ## utils.FocalLoss() #
    # 模型保存文件夹
    model_save_dir = '%s/%s' % (config.ckpt, args.model_name)#'%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # 从上一个断点，继续训练
    if args.resume:
        if os.path.exists(args.ckpt):  # 这里是存放权重的目录
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            # 如果中断点恰好为转换stage的点
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>开始训练<=========
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        # train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        # val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)

        train_loss, train_f1 = train_beat_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        val_loss, val_f1 = val_beat_epoch(model, criterion, val_dataloader)

        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_f1', val_f1, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage}
        save_ckpt(state, best_f1 < val_f1, model_save_dir)
        best_f1 = max(best_f1, val_f1)

        if val_f1 < best_f1:
            epoch_cum += 1
        else:
            epoch_cum = 0 

        # if epoch in config.stage_epoch:
        if epoch_cum == 5:
            stage += 1
            lr /= config.lr_decay
            if lr < 1e-6:
                lr = 1e-6
                print("*" * 20, "step into stage%02d lr %.3ef" % (stage, lr))
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)

        elif epoch_cum >= 8:
            print("*" * 20, "step into stage%02d lr %.3ef" % (stage, lr))
            break

######### 模型定义 #########
class MyModel(nn.Module):
    def __init__(self):   # input the dim of output fea-map of Resnet:
        super(MyModel, self).__init__()
        
        BackBone = getattr(models, config.model_name)()#models.resnet50(pretrained=True)
        
        add_block = []
        add_block += [nn.Linear(55, config.num_classes)]
        # add_block += [nn.LeakyReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_xavier)
 
        self.BackBone = BackBone
        self.add_block = add_block
 
 
    def forward(self, x, beat):   # input is 55
 
        x = self.BackBone(x, beat)
        x = self.add_block(x)
 
        return x

#python round2_beat_process.py
#python round2_data_process.py
#python main.py train_cv --model_name=mixnet_sm
def train_cv(args):
    # model
    # 模型保存文件夹
    model_save_dir = '%s/%s' % (config.ckpt, args.model_name+"_cv")#'%s/%s_%s' % (config.ckpt, args.model_name+"_cv", time.strftime("%Y%m%d%H%M"))
    for fold in range(config.kfold):
        print("fold : ",fold)
        model = getattr(models, args.model_name)()
        if args.ckpt and not args.resume:
            state = torch.load(args.ckpt, map_location='cpu')
            model.load_state_dict(state['state_dict'])
            print('train with pretrained weight val_f1', state['f1'])

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, config.num_classes)

        #2019/11/11
        #save dense/fc weight for pretrain 55 classes
        # model = MyModel()
        # num_ftrs = model.classifier.out_features
        # model.fc = nn.Linear(55, config.num_classes)

        model = model.to(device)
        # data
        train_dataset = ECGDataset(data=config.round2_train_data_cv.format(fold), 
                                data_path=config.round2_train_dir, 
                                beat_path=config.round2_beat_dir, 
                                fold=fold, 
                                train=True,
                                num_classes=34)

        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=config.batch_size, 
                                    shuffle=True, 
                                    num_workers=6)

        val_dataset = ECGDataset(data=config.round2_train_data_cv.format(fold), 
                                data_path=config.round2_train_dir, 
                                beat_path=config.round2_beat_dir, 
                                fold=fold,  
                                train=False,
                                num_classes=34)

        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=config.batch_size, 
                                    num_workers=4)

        print("fold_{}_train_datasize".format(fold), len(train_dataset), "fold_{}_val_datasize".format(fold), len(val_dataset))
        # optimizer and loss
        optimizer = radam.RAdam(model.parameters(), lr=config.pre_lr) #optim.Adam(model.parameters(), lr=config.lr)#
        w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
        criterion = utils.WeightedMultilabel(w)

        if args.ex: model_save_dir += args.ex
        best_f1 = -1
        lr = config.pre_lr
        start_epoch = 1
        stage = 1
        # 从上一个断点，继续训练
#         if args.resume:
#             if os.path.exists(args.ckpt):  # 这里是存放权重的目录
#                 model_save_dir = args.ckpt
#                 current_w = torch.load(os.path.join(args.ckpt, config.current_w))
#                 best_w = torch.load(os.path.join(model_save_dir, config.best_w))
#                 best_f1 = best_w['loss']
#                 start_epoch = current_w['epoch'] + 1
#                 lr = current_w['lr']
#                 stage = current_w['stage']
#                 model.load_state_dict(current_w['state_dict'])
#                 # 如果中断点恰好为转换stage的点
#                 if start_epoch - 1 in config.stage_epoch:
#                     stage += 1
#                     lr /= config.lr_decay
#                     utils.adjust_learning_rate(optimizer, lr)
#                     model.load_state_dict(best_w['state_dict'])
#                 print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
        logger = Logger(logdir=model_save_dir, flush_secs=2)
        # =========>开始训练<=========
        for epoch in range(start_epoch, config.max_epoch + 1):
            since = time.time()
            # train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
            # val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)

            train_loss, train_f1 = train_beat_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
            val_loss, val_f1 = val_beat_epoch(model, criterion, val_dataloader)

            print('#fold:%02d epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
                  % (fold, epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
            logger.log_value('fold{}_train_loss'.format(fold),  train_loss, step=epoch)
            logger.log_value('fold{}_train_f1'.format(fold), train_f1, step=epoch)
            logger.log_value('fold{}_val_loss'.format(fold),  val_loss, step=epoch)
            logger.log_value('fold{}_val_f1'.format(fold),  val_f1, step=epoch)
            state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                     'stage': stage}
            save_ckpt_cv(state, best_f1 < val_f1, model_save_dir,fold)
            best_f1 = max(best_f1, val_f1)

            if val_f1 < best_f1:
                epoch_cum += 1
            else:
                epoch_cum = 0 

            # if epoch in config.stage_epoch:
            if epoch_cum == 5:
                stage += 1
                lr /= config.lr_decay
                if lr < 1e-6:
                    lr = 1e-6
                    print("*" * 20, "step into stage%02d lr %.3ef" % (stage, lr))
                best_w = os.path.join(model_save_dir, config.best_w_cv.format(fold))
                model.load_state_dict(torch.load(best_w)['state_dict'])
                print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
                utils.adjust_learning_rate(optimizer, lr)

            elif epoch_cum >= 8:
                print("*" * 20, "step into stage%02d lr %.3ef" % (stage, lr))
                break

            # if epoch in config.stage_epoch:
            #     stage += 1
            #     lr /= config.lr_decay
            #     best_w = os.path.join(model_save_dir, config.best_w_cv.format(fold))
            #     model.load_state_dict(torch.load(best_w)['state_dict'])
            #     print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            #     utils.adjust_learning_rate(optimizer, lr)

def find_best_threshold():
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.externals import joblib
    from sklearn.metrics import f1_score

    test_t = []
    test_p = []
    test_pp = [0,0,0,0,0]
    for fold in range(config.kfold):
        test_t.append(np.load(os.path.join(config.val_path,"mixnet_sm_predict_fold{}_test_target.npy".format(fold))))
        test_p.append(np.load(os.path.join(config.val_path,"mixnet_sm_predict_fold{}_test_proba.npy".format(fold))))

    target = np.vstack(test_t)
    proba = np.vstack(test_p)

    print("mixnet_sm 0.5 f1_score :",utils.fbeta(target, proba > 0.5))

    best_threshold = utils.optimise_f1_thresholds_fast(target,proba,verbose=False)
    y_test = target
    y_pred_test = proba
    y_pred_thr = np.array([[1 if y_pred_test[i, j] >= best_threshold[j] else 0 for j in range(y_pred_test.shape[1])]
              for i in range(y_pred_test.shape[0])])
    print("mixnet_sm best_threshold f1_score  :", f1_score(y_test, y_pred_thr, average='micro'))

    test_t_mm = []
    test_p_mm = []
    test_pp_mm = [0,0,0,0,0]
    for fold in range(config.kfold):
        test_t_mm.append(np.load(os.path.join(config.val_path,"mixnet_mm_predict_fold{}_test_target.npy".format(fold))))
        test_p_mm.append(np.load(os.path.join(config.val_path,"mixnet_mm_predict_fold{}_test_proba.npy".format(fold))))

    target_mm = np.vstack(test_t_mm)
    proba_mm = np.vstack(test_p_mm)

    print("mixnet_mm 0.5 f1_score :",utils.fbeta(target_mm, proba_mm > 0.5))

    best_threshold = utils.optimise_f1_thresholds_fast(target_mm,proba_mm,verbose=False)
    y_test = target_mm
    y_pred_test = proba_mm
    y_pred_thr = np.array([[1 if y_pred_test[i, j] >= best_threshold[j] else 0 for j in range(y_pred_test.shape[1])]
              for i in range(y_pred_test.shape[0])])
    print("mixnet_mm best_threshold f1_score  :", f1_score(y_test, y_pred_thr, average='micro'))

    # ensemble sm+mm
    # proba_en = (proba_mm+proba)/2
    proba_en = proba
    clf = LinearRegression() #  Ridge(alpha=0)# 
    clf.fit(proba_en,target)

    joblib.dump(clf,os.path.join(args.ckpt,'mixnet_ensemble_model.pkl'))
    clf=joblib.load(os.path.join(args.ckpt,'mixnet_ensemble_model.pkl'))
        
    x_train = clf.predict(proba_en)
    print("LinearRegression ensemble 0.5 f1_score :",utils.fbeta(target,x_train > 0.5))

    best_threshold = utils.optimise_f1_thresholds_fast(target[:],x_train[:],verbose=False)
    y_test = target #[18000]
    y_pred_test = x_train #[18000]
    y_pred_thr = np.array([[1 if y_pred_test[i, j] >= best_threshold[j] else 0 for j in range(y_pred_test.shape[1])]
              for i in range(y_pred_test.shape[0])])
    print("LinearRegression ensemble  best_threshold f1_score  :", f1_score(y_test, y_pred_thr, average='micro'))

    return np.array(best_threshold)


#用于测试加载模型
def val(args):
    list_threhold = [0.5]
    threshold = list_threhold
    model = getattr(models, config.model_name)()
    if args.ckpt: model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    for threshold in list_threhold:
        val_loss, val_f1 = val_beat_epoch(model, criterion, val_dataloader, threshold,save=True)
        # print('threshold val_loss:%0.3e val_f1:%.3f\n' % (val_loss, val_f1))
        print('threshold %.2f val_loss:%0.3e val_f1:%.3f\n' % (threshold, val_loss, val_f1))

#用于测试加载模型
# python main.py val_cv --ckpt=ckpt/mixnet_sm_cv --model_name=mixnet_sm_predict
def val_cv(args):
    list_threhold = [0.5]
    threshold = list_threhold
    # model'
    kfold = 5
    model = []
    for fold in range(kfold):
        model.append(getattr(models, args.model_name)())
    for fold in range(kfold):
        model[fold].load_state_dict(torch.load(os.path.join(args.ckpt,"best_weight_fold{}.pth".format(fold)),map_location='cpu')['state_dict'])

        model[fold] = model[fold].to(device)
        model[fold].eval()
    criterion = nn.BCEWithLogitsLoss()
    # model = getattr(models, config.model_name)()
    # if args.ckpt: model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    # model = model.to(device)

    train = False
    for fold in tqdm(range(kfold)):
        output_proba = []

        val_dataset = ECGDataset(data=config.round2_train_data_cv.format(fold), 
                                data_path=config.round2_train_dir, 
                                beat_path=config.round2_beat_dir, 
                                fold=fold, 
                                train=train)

        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
        for threshold in list_threhold:
            val_loss, val_f1 = val_beat_epoch(model[fold], criterion, val_dataloader, threshold,model_name=args.model_name,save=True,fold=fold,train=train)
            # print('threshold val_loss:%0.3e val_f1:%.3f\n' % (val_loss, val_f1))
            print('threshold %.2f val_loss:%0.3e val_f1:%.3f\n' % (threshold, val_loss, val_f1))

#提交结果使用
def test(args):
    from dataset import transform
    from data_process import name2index,get_arrythmias,get_dict
    # arrythmias = get_arrythmias(config.arrythmia)
    # name2idx,idx2name = get_dict(arrythmias)
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    #utils.mkdirs(config.sub_dir)
    # model
    model = getattr(models, config.model_name)()
    print(model)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    model.eval()
    #sub_file = '%s/subB_%s.txt' % (config.sub_dir, time.strftime("%Y%m%d%H%M"))
    sub_file = 'result.txt'
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for line in tqdm(open(config.test_label, encoding='utf-8')):

            fout.write(line.strip('\n'))
            id = line.split('\t')[0]

            file_path = os.path.join(config.test_dir, id)
            df = pd.read_csv(file_path, sep=' ')
            df['III'] = df['II']-df['I']
            df['aVR'] = -(df['I']+df['II'])/2
            df['aVL'] = df['I']-df['II']/2
            df['aVF'] = df['II']-df['I']/2
            x = transform(df.values).unsqueeze(0).to(device)
            beat = transform(df.values[:300,:]).unsqueeze(0).to(device)

            output = torch.sigmoid(model(x,beat)).squeeze().cpu().numpy()
            ixs = [i for i, out in enumerate(output) if out > 0.5]
            for i in ixs:
                fout.write("\t" + idx2name[i])
            fout.write('\n')
    fout.close()

#提交结果使用
def test_cv(args):
    from dataset import transform
    from round2_data_process import name2index,get_arrythmias,get_dict
    # arrythmias = get_arrythmias(config.arrythmia)
    # name2idx,idx2name = get_dict(arrythmias)
    name2idx = name2index(config.round2_arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    #utils.mkdirs(config.sub_dir)
    num_clases = 34
    kfold = 5
    # model
    model = []
    for fold in range(kfold):
        model.append(getattr(models, config.model_name)())
    for fold in range(kfold):
        model[fold].load_state_dict(torch.load(os.path.join(args.ckpt,"best_weight_fold{}.pth".format(fold)), map_location='cpu')['state_dict'])
        model[fold] = model[fold].to(device)
        model[fold].eval()

    #sub_file = '%s/subB_%s.txt' % (config.sub_dir, time.strftime("%Y%m%d%H%M"))
    sub_file = 'result.txt'
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for line in tqdm(open(config.test_label, encoding='utf-8')):
            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.test_dir, id)
            df = pd.read_csv(file_path, sep=' ')
            df['III'] = df['II']-df['I']
            df['aVR'] = -(df['I']+df['II'])/2
            df['aVL'] = df['I']-df['II']/2
            df['aVF'] = df['II']-df['I']/2
            x = transform(df.values).unsqueeze(0).to(device)
            output = 0#np.zeros(num_clases)
            for fold in range(kfold):
                output += torch.sigmoid(model[fold](x)).squeeze().cpu().numpy()
            output = output/5
            ixs = [i for i, out in enumerate(output) if out > 0.5]
            for i in ixs:
                fout.write("\t" + idx2name[i])
            fout.write('\n')
    fout.close()

#提交结果使用
def test_ensemble(args):
    from dataset import transform
    from data_process import name2index,get_arrythmias,get_dict
    # arrythmias = get_arrythmias(config.arrythmia)
    # name2idx,idx2name = get_dict(arrythmias)
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    #utils.mkdirs(config.sub_dir)
    num_clases = 34
    kfold = len(config.model_names)
    # model
    model = []
    for fold in range(kfold):
        model.append(getattr(models, config.model_names)())
    for fold in range(kfold):
        model[fold].load_state_dict(torch.load(os.path.join(args.ckpt,config.model_ckpts[fold],"best_weight.pth"), map_location='cpu')['state_dict'])
        model[fold] = model[fold].to(device)
        model[fold].eval()

    #sub_file = '%s/subB_%s.txt' % (config.sub_dir, time.strftime("%Y%m%d%H%M"))
    sub_file = './result.txt'
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for line in tqdm(open(config.test_label, encoding='utf-8')):
            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.test_dir, id)
            df = pd.read_csv(file_path, sep=' ')
            df['III'] = df['II']-df['I']
            df['aVR'] = -(df['I']+df['II'])/2
            df['aVL'] = df['I']-df['II']/2
            df['aVF'] = df['II']-df['I']/2
            x = transform(df.values).unsqueeze(0).to(device)
            output = 0#np.zeros(num_clases)
            for fold in range(kfold):
                output += torch.sigmoid(model[fold](x)).squeeze().cpu().numpy()
            output = output/kfold
            ixs = [i for i, out in enumerate(output) if out > 0.5]
            for i in ixs:
                fout.write("\t" + idx2name[i])
            fout.write('\n')
    fout.close()
    
# python main.py test_cv_ensemble --ckpt=ckpt
def test_cv_ensemble(args):
    from dataset import transform,transform_beat
    from round2_data_process import name2index,get_arrythmias,get_dict
    from sklearn.externals import joblib

    # arrythmias = get_arrythmias(config.round1_arrythmia)
    # name2idx,idx2name = get_dict(arrythmias)
    name2idx = name2index(config.round2_arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}

    num_clases = 34
    kfold_sm = 5
    kfold_mm = 5

    # model
    model_sm = []
    for fold in range(kfold_sm):
        model_sm.append(getattr(models, config.model_names[0])())
    for fold in range(kfold_sm):
        model_sm[fold].load_state_dict(torch.load(os.path.join(config.ckpt,config.model_ckpts[0],"best_weight_fold{}.pth".format(fold)), map_location='cpu')['state_dict'])
        model_sm[fold] = model_sm[fold].to(device)
        model_sm[fold].eval()

    model_mm = []
    for fold in range(kfold_mm):
        model_mm.append(getattr(models, config.model_names[1])())
    for fold in range(kfold_mm):
        model_mm[fold].load_state_dict(torch.load(os.path.join(config.ckpt,config.model_ckpts[1],"best_weight_fold{}.pth".format(fold)), map_location='cpu')['state_dict'])
        model_mm[fold] = model_mm[fold].to(device)
        model_mm[fold].eval()

    threshold = find_best_threshold()

    model_lr = joblib.load(os.path.join(config.ckpt,'mixnet_ensemble_model.pkl'))


    sub_file = './result.txt'

    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for line in tqdm(open(config.round2_test_label, encoding='utf-8')):
            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.round2_test_dir, id)
            df = pd.read_csv(file_path, sep=' ')

            file_beat_path = os.path.join(config.round2_test_beat_dir,id.split(".")[0]+"_beat.txt")
            df_beat = pd.read_csv(file_beat_path, sep=' ')
            beat = transform_beat(df_beat.values).unsqueeze(0).to(device)

            df['III'] = df['II']-df['I']
            df['aVR'] = -(df['I']+df['II'])/2
            df['aVL'] = df['I']-df['II']/2
            df['aVF'] = df['II']-df['I']/2
            x = transform(df.values).unsqueeze(0).to(device)

            output_sm = 0#np.zeros(num_clases)
            for fold in range(kfold_sm):
                output_sm += torch.sigmoid(model_sm[fold](x,beat)).squeeze().cpu().numpy()
            output_sm = output_sm/kfold_sm

            output_mm = 0
            for fold in range(kfold_mm):
                output_mm += torch.sigmoid(model_mm[fold](x,beat)).squeeze().cpu().numpy()
            output_mm = output_mm/kfold_mm

            output = (output_sm + output_mm)/2

            outout = model_lr.predict(output.reshape(1,-1))

            ixs = [i for i, out in enumerate(output) if out > threshold[i]]

            for i in ixs:
                fout.write("\t" + idx2name[i])
            fout.write('\n')
    fout.close()

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--model_name", type=str, help="the model name")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "train"):
        train(args)
    if (args.command == "test_cv_ensemble"):
        test_cv_ensemble(args)
    if (args.command == "train_cv"):
        train_cv(args)
    if (args.command == "test"):
        test(args)
    if (args.command == "test_cv"):
        test_cv(args)
    if (args.command == "val_cv"):
        val_cv(args)
    if (args.command == "val"):
        val(args)
