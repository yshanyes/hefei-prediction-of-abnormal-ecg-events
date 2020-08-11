import wfdb.processing as wp
from scipy import signal
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

def load_data(case):#定义更多参数，具体参考官方说明
    df = pd.read_csv(case,sep=" ")
    df['III'] = df['II']-df['I']
    df['aVR']=-(df['II']+df['I'])/2
    df['aVL']=(df['I']-df['II'])/2
    df['aVF']=(df['II']-df['I'])/2
    return df 

def qrs_det(df, col,fs=500,max_bpm=230):
    search_radius = int(fs * 60 / max_bpm)
    p_signal = signal.resample(df[col]*0.00488, fs*10)
    qrs_inds = wp.gqrs_detect(p_signal, fs)
    
    if len(qrs_inds) == 0:
        return p_signal, []
    
    corrected_peak_inds = wp.correct_peaks(p_signal, 
                                           peak_inds=qrs_inds,
                                           search_radius=search_radius, 
                                           smooth_window_size=150)
    
    return sorted(corrected_peak_inds)

def extract_beat(p_signal, peak_inds,fs=500):
    ms300 = int(0.25*fs)#+1
    ms400 = int(0.35*fs)#+1
    peak_inds=sorted(peak_inds)

    beat_mat=[]
    for i,ind in enumerate(peak_inds):
        ind_begin = ind - ms300;
        ind_end = ind + ms400;
        
        if (ind_begin<0 or ind_end>len(p_signal)):
            pass
        else:
            if beat_mat==[]:
                beat_mat = p_signal[ind_begin:ind_end]
            else:
                beat = p_signal[ind_begin:ind_end]
                beat_mat = np.row_stack((beat_mat, beat)) 
    if (beat_mat.ndim == 1):       
        return np.round(beat_mat,decimals=2)
    else:       
        return np.round(np.median(beat_mat, axis=0),decimals=2)

def extract_median_beat(path,beat_path,fs=500):
    ms300 = int(0.25*fs)#+1
    ms400 = int(0.35*fs)#+1
    recordname=os.listdir(path)
    recordname.sort()
    hr_vec = np.zeros(len(recordname))
    for i,value in tqdm(enumerate(recordname)):
        try:
            if os.path.isfile(os.path.join(path,value)):
                df = load_data(os.path.join(path,value))

                peak_inds=[]
                for col in df.columns:
                    peak_inds = qrs_det(df, col)
                    if (len(peak_inds)>2):
                        break
                    else:
                        pass

                if (len(peak_inds)>2):
                    pass
                else:
                    p_signal = signal.resample(df['II'], fs*10)
                    peak_inds = np.where(p_signal[ms300:-ms400]==np.max(p_signal[ms300:-ms400]))[0]+ms300
                    print(value+":未检测到R波")

                beat_mat=[]
                for col in df.columns:
                    p_signal = signal.resample(df[col], fs*10)
                    beat_singlelead = extract_beat(p_signal, peak_inds)

                    if (col == 'I'):
                        beat_mat = beat_singlelead
                    else:
                        beat_mat = np.row_stack((beat_mat, beat_singlelead))

                df_new = pd.DataFrame(beat_mat.T)
                df_new.columns = df.columns
                #print(df_new.shape)
                df_new.to_csv(os.path.join(beat_path,value[0:-4]+"_beat.txt"), index=False, sep=" ")
            else:
                pass
        except Exception as e:
            print('Error:',e)
            df = load_data(os.path.join(path,value))
            peak_inds = np.where(p_signal[ms300:-ms400]==np.max(p_signal[ms300:-ms400]))[0]+ms300
            df_new = pd.DataFrame(df.values[peak_inds[0]-150:peak_inds[0]+150,:],columns=df.columns)
            #print(df_new.shape)
            df_new.to_csv(os.path.join(beat_path,value[0:-4]+"_beat.txt"), index=False, sep=" ")

if __name__ == '__main__':
    from config import config

    base_train_dir = config.round1_train_dir #r"./tcdata/train/"
    base_testA_dir = config.round1_test_dir #"./tcdata/testA/"
    beat_dir = config.round1_beat_dir #r"./tcdata/hf_round1_beat_500/"

    # if not os.path.exists(beat_dir): 
    #     os.makedirs(beat_dir) 

    extract_median_beat(base_train_dir,beat_dir)
    extract_median_beat(base_testA_dir,beat_dir)