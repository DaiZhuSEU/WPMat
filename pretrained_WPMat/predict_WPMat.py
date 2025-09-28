

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns

import random

max_len = 33
max_mid = (max_len ) // 2


def iteration_WTM():
    global pep, tcr, y_real, max_len, WTM, WTM_1d, c_map, residue_inter
    
    c_map = [[0] *(max_len *max_len) for i in range(len(pep))]
    cmap_store = [[0]*max_len for i in range(max_len)]

    for k in range(len(pep)):

        cmap_store = [[0]*max_len for i in range(max_len)]
        for i in range(len(pep[k])):
            pep_mid = (len(pep[k]) ) // 2
            for j in range(len(tcr[k])):
                tcr_mid = (len(tcr[k]) ) // 2
                #cmap_store[max_mid + i - pep_mid][max_mid + j - tcr_mid] = random.random()
                cmap_store[max_mid + i*2 - len(pep[k]) + 1][max_mid + j*2 - len(tcr[k]) + 1] = residue_inter[ord(pep[k][i]) -65][ord(tcr[k][j]) -65]
                
        c_map[k] = [item for sublist in cmap_store for item in sublist]
        
    A = np.array(c_map)
    B = np.array(y_real)

    xx = np.linalg.pinv(A)@ B      
    xx_2d = xx.reshape(max_len, max_len)

    WTM = [[0] *max_len for i in range(max_len)]

    WTM = np.array(xx_2d)
    '''
    plt.figure()
    plt.imshow(WTM)
    plt.title('WTM')
    plt.show()
    
    plt.figure()
    plt.imshow(WTM_0)
    plt.title('WTM_0')
    plt.show()
    '''
    
    WTM = (WTM + WTM_0 / 0.1) / 2

    WTM_1d = np.array(WTM)
    WTM_1d = WTM_1d.reshape(-1, 1)


def iteration_rrintermap():
    global pep, tcr, y_real, max_len, WTM, WTM_1d, c_map, residue_inter

    wt_total_1d = [[0] *26*26 for i in range(len(pep))]
    for k in range(len(pep)):

        wt_total = [[0] *26 for i in range(26)]
        for i in range(len(pep[k])):
            pep_mid = (len(pep[k]) ) // 2
            for j in range(len(tcr[k])):
                tcr_mid = (len(tcr[k]) ) // 2

                wt_total[ord(pep[k][i]) -65][ord(tcr[k][j]) -65] = wt_total[ord(pep[k][i]) -65][ord(tcr[k][j]) -65] + WTM[max_mid + i*2 - len(pep[k]) + 1][max_mid + j*2 - len(tcr[k]) + 1]
                wt_total[ord(tcr[k][j]) -65][ord(pep[k][i]) -65] = wt_total[ord(tcr[k][j]) -65][ord(pep[k][i]) -65] + WTM[max_mid + i*2 - len(pep[k]) + 1][max_mid + j*2 - len(tcr[k]) + 1]
                
                #wt_total[ord(tcr[k][j]) -65][ord(pep[k][i]) -65] = wt_total[ord(pep[k][i]) -65][ord(tcr[k][j]) -65]

        wt_total_1d[k] = [item for sublist in wt_total for item in sublist]

    wt_total_1d = np.array(wt_total_1d)
    y_real = np.array(y_real)

    rr_inter_1d = np.linalg.pinv(wt_total_1d) @ y_real

    residue_inter = rr_inter_1d.reshape(26, -1)
    residue_inter = (residue_inter + residue_inter_0 / 4000) / 2


def calculate_cmap():
    global pep, tcr, y_real, max_len, WTM, WTM_1d, c_map, residue_inter
    c_map = [[0] *(max_len *max_len) for i in range(len(pep))]
    cmap_store = [[0]*max_len for i in range(max_len)]

    for k in range(len(pep)):
        cmap_store = [[0]*max_len for i in range(max_len)]
        for i in range(len(pep[k])):
            pep_mid = (len(pep[k]) ) // 2
            for j in range(len(tcr[k])):
                tcr_mid = (len(tcr[k]) ) // 2
                #cmap_store[max_mid + i - pep_mid][max_mid + j - tcr_mid] = random.random()
                cmap_store[max_mid + i*2 - len(pep[k]) + 1][max_mid + j*2 - len(tcr[k]) + 1] = residue_inter[ord(pep[k][i]) -65][ord(tcr[k][j]) -65]
                
        c_map[k] = [item for sublist in cmap_store for item in sublist]


def data_filter(data):
    data['cdr3'] = ['C' + item + 'F' for item in data['cdr3']]
    cdr3_len = pd.Series([len(item) for item in data['cdr3']])
    #data = data[cdr3_len == 13]
    '''
    mutation_point = 13
    
    data['cdr3'] = [item[0 : mutation_point] + item[mutation_point + 1 : len(item)] for item in data['cdr3']]
    '''
    
    
    delet_len = 4
    data['cdr3'] = [item[delet_len : len(item) - delet_len] for item in data['cdr3']]
    
    
    '''
    start_point = 9
    keep_len = 4
    data['cdr3'] = [item[start_point : start_point + keep_len] for item in data['cdr3']]
    print(data['cdr3'])
    '''
    
    '''
    start_point = 12
    delet_len = 1
    data['cdr3'] = [item[0 : start_point] + item[start_point + delet_len : len(item)] for item in data['cdr3']]
    '''
    return data



##############测试#####################


WTM_input = pd.read_csv('parameter/PAWM.mat')
WTM_input = np.array(WTM_input)
WTM_1d = WTM_input.reshape(-1, 1)
residue_inter = pd.read_csv('parameter/RRIPM.mat')
residue_inter = np.array(residue_inter)

data_read = pd.read_csv('data_input/data_test.csv')
data_read_raw = data_read

data_read = data_filter(data_read)

pep = data_read['peptide'].tolist()
tcr = data_read['cdr3'].tolist()
#tcr = [item[3 : len(item) - 3] for item in tcr]
y_real = data_read['bind'].tolist()

calculate_cmap()
                            
y_predict = c_map[0:][:] @ WTM_1d 
y_predict = (y_predict - y_predict.mean() / 2) / (y_predict.mean() / 2) / 2
data_read_raw['predicted_score'] = y_predict

data_read_raw.to_csv('output/prediction_output.csv', index = None)
