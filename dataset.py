# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import os, copy,pywt
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
from scipy import signal as sig
from scipy.signal import medfilt
import warnings
warnings.filterwarnings('ignore')

def wavelet_sym(df_data, wfun='sym8', dcmp_levels=8, chop_levels=3, fs=256, 
    removebaseline=False, normalize=False):

    for i in range(df_data.shape[1]):
        data = df_data[:,i]
        data = butterworth_notch(data, cut_off = [49, 51], order = 2, sampling_freq = fs)

        if removebaseline:
            first_filtered = medfilt(data,71)
            second_filtered = medfilt(first_filtered,215)
            data = data - second_filtered

        dcmp_levels = min(dcmp_levels, pywt.dwt_max_level(data.shape[0], pywt.Wavelet(wfun)))

        coeffs = pywt.wavedec(data, wfun, mode='symmetric', level = dcmp_levels, axis = -1)
        #
        coeffs_m = [np.zeros_like(coeffs[idx]) if idx >= -chop_levels  else coeffs[idx] for idx in range(-dcmp_levels- 1, 0)]
        #
        df_data[:,i] = pywt.waverec(coeffs_m, wfun, mode='symmetric', axis = -1)
        #
        if normalize:
            df_data[:,i] = (df_data[:,i] - np.mean(df_data[:,i])) /np.std(df_data[:,i])
        #data_recon = butterworth_high_pass(data_recon, cut_off = 0.5, order = 6, sampling_freq = fs)#cut_off=2
        #data_recon = butterworth_notch(data_recon, cut_off = [49, 51], order = 2, sampling_freq = fs)
    return df_data

def butterworth_high_pass(x, cut_off, order, sampling_freq):
    #
    nyq_freq = sampling_freq / 2
    digital_cutoff = cut_off / nyq_freq
    #
    b, a = sig.butter(order, digital_cutoff, btype='highpass')
    y = sig.lfilter(b, a, x, axis = -1)
    
    return y

def butterworth_notch(x, cut_off, order, sampling_freq):
    #
    cut_off = np.array(cut_off)
    nyq_freq = sampling_freq / 2
    digital_cutoff = cut_off / nyq_freq
    #
    b, a = sig.butter(order, digital_cutoff, btype='bandstop')
    y = sig.lfilter(b, a, x, axis = -1)
    #
    return y

def wavelet_db4(df_data, wavefunc="db4", lv=4, m=2, n=4):  #

    for i in range(df_data.shape[1]):
        data = df_data[:,i]
        coeff = pywt.wavedec(data, wavefunc, level=lv)  #mode='db', 
        # sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0

        for i in range(m, n + 1):
            cD = coeff[i]
            for j in range(len(cD)):
                Tr = np.sqrt(2 * np.log(len(cD)))
                if cD[j] >= Tr:
                    coeff[i][j] = np.sign(cD[j]) - Tr
                else:
                    coeff[i][j] = 0

        df_data[:,i] = pywt.waverec(coeff, wavefunc)
    return df_data

def wavelet_db6(df_data,wavefunc="db6", lv=8):
    """
    R J, Acharya U R, Min L C. ECG beat classification using PCA, LDA, ICA and discrete
     wavelet transform[J].Biomedical Signal Processing and Control, 2013, 8(5): 437-448.
    param sig: 1-D numpy Array
    return: 1-D numpy Array
    """
    ecg_sig = np.zeros([df_data.shape[0],df_data.shape[1]])
    for i in range(df_data.shape[1]):
        data = df_data[:,i]
        coeffs = pywt.wavedec(data, wavefunc, level=lv)
        coeffs[-1] = np.zeros(len(coeffs[-1]))
        coeffs[-2] = np.zeros(len(coeffs[-2]))
        coeffs[0] = np.zeros(len(coeffs[0]))
        ecg_sig[:,i] = pywt.waverec(coeffs, wavefunc)
    return ecg_sig

from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, lfilter, lfilter_zi, filtfilt, sosfreqz, resample
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def butter_bandpass_filter(df_data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    ecg_sig = np.zeros([df_data.shape[0],df_data.shape[1]])
    for i in range(df_data.shape[1]):
        data = df_data[:,i]
        y = sosfilt(sos,
                    data)  # Filter data along one dimension using cascaded second-order sections. Using lfilter for each second-order section.
        ecg_sig[:,i] = y
    return ecg_sig

def butter_bandpass_forward_backward_filter(df_data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    ecg_sig = np.zeros([df_data.shape[0],df_data.shape[1]])
    for i in range(df_data.shape[1]):
        data = df_data[:,i]
        y = sosfiltfilt(sos,
                        data)  # Apply a digital filter forward and backward to a signal.This function applies a linear digital filter twice, once forward and once backwards. The combined filter has zero phase and a filter order twice that of the original.
        ecg_sig[:,i] = y
    return ecg_sig

def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def verflip(sig):
    '''
    信号竖直翻转
    :param sig:
    :return:
    '''
    return sig[::-1, :]

def shift(sig, interval=20):
    '''
    上下平移
    :param sig:
    :return:
    '''
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig

#https://www.physionet.org/content/nstdb/1.0.0/
def transform(sig, train=True):
    # 前置不可或缺的步骤
    sig = resample(sig, config.target_point_num)
    # # 数据增强
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.3: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
        # if np.random.randn() > -1: 
        #     fi = np.random.randint(11)
        #     if fi % 2 == 0 and fi != 2 and fi != 0 :
        #         sig = wavelet_db6(sig,'db{}'.format(fi) ,8)
        #     else:#if  fi % 2 != 0:
        #         if np.random.randn() > -0.5:
        #             sig = butter_bandpass_filter(sig,0.05,40,256)
        #         else:
        #             sig = butter_bandpass_forward_backward_filter(sig,0.05,40,256)
    else:
        # sig = butter_bandpass_filter(sig,0.05,40,256)
        pass
    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig

def transform_beat(sig, train=False):
    # 前置不可或缺的步骤
    # sig = resample(sig, config.target_point_num)
    # # 数据增强
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
        # if np.random.randn() > 0.4: sig = wavelet_db6(sig)
        # if np.random.randn() > 0.5: sig = wavelet_db4(sig) # time consuming
        # if np.random.randn() > 0.3: sig = wavelet_sym(sig)

    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """
    def __init__(self, data, data_path, beat_path, num_classes=34,fold=0, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(data)   #config.train_data_cv.format(fold) #train_resample_data
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])

        self.data_path   = data_path
        self.beat_path   = beat_path
        self.num_classes = num_classes

    def __getitem__(self, index):
        fid = self.data[index]

        file_path = os.path.join(self.data_path, fid)# if self.train else os.path.join(config.trainA_dir, fid)
        df = pd.read_csv(file_path, sep=' ')#.values

        file_beat_path = os.path.join(self.beat_path,fid.split(".")[0]+"_beat.txt")
        df_beat = pd.read_csv(file_beat_path, sep=' ')
        
        df['III'] = df['II']-df['I']
        df['aVR'] = -(df['I']+df['II'])/2
        df['aVL'] = df['I']-df['II']/2
        df['aVF'] = df['II']-df['I']/2

        x = transform(df.values, self.train)
        beat = transform_beat(df_beat.values, self.train)

        target = np.zeros(self.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, beat, target #

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    print(d[0])
