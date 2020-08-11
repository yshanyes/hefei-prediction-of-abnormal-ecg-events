# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
class FocalLoss2d1(nn.Module):
    def __init__(self, gamma=2, class_weight=None, size_average=True):
        super(FocalLoss2d1, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.class_weight = class_weight

    def forward(self, logit, target, type='sigmoid'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if self.class_weight is None:
                self.class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()
            if self.class_weight is None:
                self.class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(self.class_weight).cuda().view(-1,1)
        class_weight = torch.gather(self.class_weight, 0, target)
        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss

class FocalLoss1(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')#reduction='none', reduce=True
        self.focusing_param = focusing_param
        self.balance_param = balance_param
        self.size_average = True

    def forward(self, output, target):
        # print(output)
        # print(target)
        logpt = self.cerition(output, target)
        # cross_entropy = F.cross_entropy(output, target)
        # cross_entropy_log = torch.log(cross_entropy)
        # logpt = - F.cross_entropy(output, target)
        pt    = torch.exp(logpt)
        focal_loss = ((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        if self.size_average:
            loss = balanced_focal_loss.mean()
        else:
            loss = balanced_focal_loss

        return loss

class FocalLoss(nn.Module):

    def __init__(self, gama=10, alpha=0.5, size_average =True):
        super(FocalLoss, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')#reduction='none', reduce=True
        self.gama = gama
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, output, target):
        #logpt = - F.binary_cross_entropy_with_logits(output, target,reduction='mean')#self.cerition(output, target)
        #pt    = torch.exp(logpt)
        p = output.sigmoid()

        focal_loss = -self.alpha*(1-p)**self.gama * p.log()*target - (1-self.alpha)*(p)**self.gama * (1-p).log()*(1-target) #.mean()

        #focal_loss = -((1 - pt) ** self.gama) * logpt
        #balanced_focal_loss = self.balance_param * focal_loss

        if self.size_average:
            loss = focal_loss.mean()
        else:
            loss = focal_loss.sum()

        loss = Variable(loss, requires_grad = True)

        return loss

class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        weight = Variable(self.weight)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss


        return balanced_focal_loss


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)

# 计算F1score
def re_calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.cpu().detach().numpy().astype(np.int)
    # print(y_true.shape)
    y_prob = y_pre.cpu().detach().numpy()
    y_pre = y_prob > threshold #* (y_true.shape[0]//34)).astype(np.int)
    return y_true, y_prob, f1_score(y_true, y_pre,average='micro')
    
def fbeta(true_label, prediction):
    from sklearn.metrics import f1_score
    return f1_score(true_label, prediction, average='micro')#'micro', 'macro', 'weighted', 'samples'

def optimise_f1_thresholds_fast(y, p, iterations=20, verbose=True,num_classes=34):
    best_threshold = [0.2]*num_classes
    for t in range(num_classes):
        best_fbeta = 0
        temp_threshhold = [0.2]*num_classes
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(y, p > temp_threshhold)
            if temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshold[t] = temp_value

        if verbose:
            print(t, best_fbeta, best_threshold[t])

    return best_threshold
#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()
