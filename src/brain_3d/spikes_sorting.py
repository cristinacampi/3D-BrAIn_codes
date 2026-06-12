"""
Codes with functions related to Spikes sorting
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pywt
import math
import scipy
import bxr_functions
import brw_functions as brw_f
import time
from scipy.signal import find_peaks, butter, filtfilt
from statistics import median
import stratification
from scipy.stats import pearsonr
import igraph as ig
from igraph import Graph, plot
import leidenalg as la
import random
import merging_tree as merge
import matplotlib.patches as mpatches
from neo import SpikeTrain 
import quantities as pq
from elephant.spike_train_correlation import cross_correlation_histogram
from concurrent.futures import ProcessPoolExecutor
import psutil
import os
import gc

def FindCorrelation(df, thresh=0.9, verbose=False):
    """
    _summary_
    
    Args:
        df (_type_): _description_.
        thresh (float): _description_. Defaults to 0.9.
        verbose (bool, optional): _description_. Defaults to False. Raises:. ValueError: _description_.
    
    Returns:
        _type_: _description_.
    """
    corrMatrix = df.corr()
    # corrMatrix.loc[:,:] =  np.triu(corrMatrix, k=0)

    varnum = corrMatrix.shape[0]
    
    if varnum == 1:
        raise ValueError("only one variable given")
        
    # Re-order columns based on max absolute correlation
    '''
    original_order = np.arange(varnum)
    diag_mask = np.eye(varnum, dtype=bool)
    corrMatrix[diag_mask] = np.nan
    max_abs_corr_order = np.argsort(np.nanmax(np.abs(corrMatrix), axis=0))[::-1]
    corrMatrix = corrMatrix.iloc[:, max_abs_corr_order]
    new_order = original_order[max_abs_corr_order]
    temp_matrix = corrMatrix.copy()
    '''

    # Order columns based on average correlation
    original_order = np.arange(varnum)
    diag_mask = np.eye(varnum, dtype=bool)
    corrMatrix[diag_mask] = np.nan
    max_abs_corr_order = np.argsort(np.nanmax(-np.abs(corrMatrix), axis=0))
    corrMatrix = corrMatrix.iloc[:, max_abs_corr_order]
    new_order = original_order[max_abs_corr_order]
    mean_abs_corr_order = np.argsort(np.nanmean(-np.abs(corrMatrix), axis=0))
    corrMatrix = corrMatrix.iloc[:, mean_abs_corr_order]
    new_order = new_order[mean_abs_corr_order]
    temp_matrix = (corrMatrix.copy())
    #temp_matrix[diag_mask] = np.nan
    #'''

    delete_col = list(original_order)
    original_order = list(original_order)
    new_order = list(new_order)
    col = []
    cont = 0
    while np.any(temp_matrix[~np.isnan(temp_matrix)] > thresh) and len(new_order)>0:
        cont += 1
        t=time.time()
        # print('cycle n°:' +str(cont))
        if verbose:
            print("All correlations <=", thresh)
            break
        idx = np.where(np.array(temp_matrix[new_order[0]])>thresh)[0]
        for i in range(len(idx)):
            delete_col.remove(original_order[idx[i]])
            new_order.remove(original_order[idx[i]])
        col.append(new_order[0])
        delete_col.remove(new_order[0])
        new_order.remove(new_order[0])
        original_order = delete_col.copy()
        temp_matrix = temp_matrix[new_order]
        temp_matrix = temp_matrix.loc[delete_col]
        # print(str(time.time()-t))

    if temp_matrix.shape[0]>0:
        for i in range(temp_matrix.shape[0]):
            col.append(temp_matrix.columns[i])

    return sorted(col)


def SpikesDetection(data, step, threshold, aux_spike): 
    """
    Spikes detection on negative peaks: we have a spike when the peak is lower than a threshold t = -mu-threshold*sigma (mu is the mean of the signal and sigma is the standard deviation)
    
    Returns:
        frames (np.ndarray): list of frames indexes where a negative spike is detected.
    """
    data = data[:data.shape[0] - (data.shape[0] % step)]
    data_reshaped = data.reshape(-1, step)
    mu = np.mean(data_reshaped,axis=1)
    sigma = np.std(data_reshaped,axis=1)
    data_reshaped_aux = data_reshaped-mu[:, np.newaxis] 
    th_sigma = threshold*sigma
    
    frames = []  # per salvare gli indici dei picchi per ogni riga
    for ii in range(data_reshaped_aux.shape[0]):
        row = data_reshaped_aux[ii, :]
        if aux_spike == "pos":
            peaks, properties = find_peaks(row, height=th_sigma[ii])
        else:
            peaks, properties = find_peaks(-row, height=th_sigma[ii])
        if len(peaks) > 0:
            peaks = peaks[np.argmax(properties["peak_heights"])]
            peaks  = peaks + step*ii
            frames.append(peaks)
    #if len(frames)>0:
    #    frames = np.concatenate(frames)
    #    frames = np.sort(frames)

    return frames

def TemplateNeg(data, ch, parameter = 4.5, algo = 'Leiden', distance = 'rho', method_HC = 'complete', criterion_HC = 'distance', method_KM = 'silhouette', max_iter_FCM=10, threshold_variance = 0.9, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, fuzzy_parameter = 1, noise = 0, threshold_dendrogram = 0.33, max_classes = [2], threshold_Leiden = 0.9, p_minkowski = 2, frequency=1000, normalization = 'OFF', norm_mode ='min_max_single'):
    """
    Negative templates learning
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset (number of frames x number of channels).
        ch (int): Channel idx (between 0 and number of channels -1) on which we want to learn templates.
        parameter (float): spikes detection parameter. Defaults to 4.5.
        algo (str): clustering algorithm'. Defaults to 'Leiden'.
        distance (str): metric for clustering. Defaults to 'rho'.
        method_HC (str): linkage method. Defaults to 'complete'.
        criterion_HC (str): hierarchical clustering criterion. Defaults to 'distance'.
        method_KM (str): method to compute the optimal number of centroids in KM and FCM or relatives. Defaults to 'silhouette'.
        max_iter_FCM (int): maximum number of iterations in FCM. Defaults to 10.
        threshold_variance (float): explained variance after PCA. Defaults to 0.9.
        w_max (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilon_EDR (float): EDR threshold. Defaults to 0.001.
        epsilon_LCSS (float): LCSS threshold. Defaults to 0.001.
        fuzzy_parameter (int): FCM parameter. Defaults to 1.
        noise (int): amount of noise in percentage to add to data. Defaults to 0.
        threshold_dendrogram (float): cut height of the dendrogram. Defaults to 0.33.
        max_classes (int): maximum number of classes for clustering. Defaults to 1.
        threshold_Leiden (float): Leiden threshold. Defaults to 0.9.
        p_minkowski (int): Minkowski parameter. Defaults to 2.
        frequency (int): STS parameter. Defaults to 1000.
        normalization (str): To applying normalization. Defaults to 'OFF'.
        norm_mode (str): If normalization applied, to select the modality. Defaults to 'min_max_single'.
    
    Returns:
        clusters (list): number of cluster and clusters obtained from template learning.
        templates_N (np.ndarray): centroids of the clusters.
        frames_N (list): frames idx of all the spikes detected.
    """

    DataChannel = data[:, ch].copy()
    NumFrames = DataChannel.shape[0] 
    step = int(frequency*0.01) 
    t=0
    n_frames_ch_neg = 0
    frames_N = {}  
    while(t<NumFrames-step):
        mu = np.mean(data[t:t+step, ch])
        sigma = np.std(data[t:t+step, ch]) 
        frames_neg = SpikesDetectionNeg(data[t:t+step,:] , ch, parameter)+t
        t=t+step
        n_frames_ch_neg += len(frames_neg)
        frames_N = set(frames_N)|set(frames_neg)
    mu = np.mean(data[t:NumFrames, ch])
    sigma = np.std(data[t:NumFrames, ch]) 
    frames_neg = SpikesDetectionNeg(data[t:NumFrames,:] , ch, parameter)+t
    n_frames_ch_neg += len(frames_neg)
    frames_N = set(frames_N)|set(frames_neg)
    frames_N = sorted(frames_N)

    #PROVA PER MIGLIORARE MA DA RICONTROLLARE
    #'''
    t=0
    n_frames_ch_pos = 0
    frames_P = {}  
    while(t<=NumFrames-step):
        mu = np.mean(data[t:t+step, ch])
        sigma = np.std(data[t:t+step, ch]) 
        frames_pos = SpikesDetectionNeg(-data[t:t+step,:] , ch, 4.5)+t
        t=t+step
        n_frames_ch_pos += len(frames_pos)
        frames_P = set(frames_P)|set(frames_pos)
    mu = np.mean(data[t:NumFrames, ch])
    sigma = np.std(data[t:NumFrames, ch]) 
    frames_pos = SpikesDetectionNeg(-data[t:NumFrames,:] , ch, 4.5)+t
    n_frames_ch_pos += len(frames_pos)
    frames_P = set(frames_P)|set(frames_pos)
    frames_P = sorted(frames_P)
    frames_N_new = set(frames_N)
    l=0
    while l < len(frames_P):
        if len(set(np.array(range(frames_P[l]-5, frames_P[l]))) & set(frames_N))>0:
            f = sorted(set(np.array(range(frames_P[l]-5, frames_P[l]))) & set(frames_N))[-1]
            if DataChannel[frames_P[l]]>-DataChannel[f]:
                frames_N_new.remove(f)
            l=l+1
        elif  len(set(np.array(range(frames_P[l], frames_P[l]+5+1))) & set(frames_N))>0:
            f = sorted(set(np.array(range(frames_P[l], frames_P[l]+5+1))) & set(frames_N))[0]
            if DataChannel[frames_P[l]]>-DataChannel[f]:
                frames_N_new.remove(f)
            l=l+1
        else:
            l=l+1
    frames_N = sorted(frames_N_new)
    #'''

    dataset_N = np.zeros((len(frames_N), 41)) 
    for k in range(len(frames_N)):
        peak_frame = frames_N[k] 
        if peak_frame < 20:
            dataset_N[k, 20-peak_frame:41] = DataChannel[0:peak_frame+21] 
        elif peak_frame >= NumFrames-20:
            dataset_N[k, 0: NumFrames-peak_frame+20] = DataChannel[peak_frame-20:NumFrames] 
        else:
            dataset_N[k] = DataChannel[peak_frame-20:peak_frame+21] 
    dataset_N_aux = dataset_N.copy()
    if dataset_N_aux.shape[0]>1:
        clusters = stratification_sktime.Recursive_clustering(data=dataset_N_aux, algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, SamplingRate=frequency, p_minkowski=p_minkowski, normalization=normalization, norm_mode=norm_mode)

        templates_N =[] 
        for c in range(clusters[0]):
            data = dataset_N[clusters[1][c]]
            data = data.reshape((data.shape[-2], data.shape[-1]))
            mu = np.mean(data,0)
            templates_N.append(mu)
        templates_N = np.array(templates_N).T
        # '''
        df = pd.DataFrame(templates_N)
        corr = np.array(df.corr())-np.eye(templates_N.shape[1])
        idxs = set(np.arange(templates_N.shape[1]))
        while (np.max(corr)>=0.95): 
            idxs_del = np.where(corr==np.max(corr))
            a = idxs_del[0][0]
            b = idxs_del[1][0]
            idxs = idxs-{a,b} 
            clusters_new =[]
            for i in idxs:
                clusters_new.append(clusters[1][i])
            clusters_new.append(list(set(clusters[1][a])|set(clusters[1][b])))
            clusters = (len(clusters_new), clusters_new)
            templates_N =[] 
            for c in range(clusters[0]):
                data = dataset_N[clusters[1][c]]
                data = data.reshape((data.shape[-2], data.shape[-1]))
                mu = np.mean(data,0)
                templates_N.append(mu)
            templates_N = np.array(templates_N).T
            df = pd.DataFrame(templates_N)
            corr = np.array(df.corr())-np.eye(templates_N.shape[1])
            idxs = set(np.arange(templates_N.shape[1]))
            # '''
                
        '''
        plt.figure()
        for c in range(clusters[0]):
            # plt.figure()
            data = dataset_N[clusters[1][c]]
            mu = np.mean(data,0)
            sigma = np.std(data,0) 
            plt.plot(np.arange(data.shape[1]), mu)
            plt.fill_between(np.arange(data.shape[1]), mu-sigma, mu+sigma, alpha = 0.2)
            # for k in range(len(clusters[1][c])):
                # plt.plot(np.arange(41), dataset_N[clusters[1][c][k]]) 
            # plt.savefig('Cluster_'+str(c))
        plt.ylabel('(uV)')
        plt.xlabel('Frames')
        plt.title('3000 cells seeded, Day 97, chip 1003, channel 1210\nTemplates (PCA&k-means algo)')
        plt.savefig('Clusters_N')
        # '''

        clusters_N = [] 
        templates = []  
        frames_new = ()
        dt = 1/frequency
        for c in range(clusters[0]):
            dy = np.diff(templates_N[:,c])/dt
            dy = np.concatenate(([0], dy)) 
            i_peaks = scipy.signal.find_peaks(-templates_N[:,c], width = 2.5)[0] #lavorare un po' qua
            # i_peaks = scipy.signal.find_peaks(-dy, height=200)[0]
            der = dy[i_peaks] 
            # if len(scipy.signal.find_peaks(-templates_N[:,c], height = -(np.mean(templates_N[:,c])-0.5*np.std(templates_N[:,c])))[0])==1:
            #if len(der[der<-50])==1 and len(scipy.signal.find_peaks(-templates_N[:,c], height = -(np.mean(templates_N[:,c])))[0])==1 and len(clusters[1][c])>2:
            if len(scipy.signal.find_peaks(-templates_N[:,c], prominence=4)[0])==1 and len(clusters[1][c])>2 and len(scipy.signal.find_peaks(templates_N[:,c], prominence=25)[0])<=1:
            # if len(clusters[1][c])>=len(frames_N)/25 and len(clusters[1][c])>=50:
            # if len(np.where(templates_N[i_peaks,c]<0)[0])==1: 
                clusters_N.append(clusters[1][c])
                templates.append(templates_N[:, c])
                frames_new = set(frames_new)|set(np.array(frames_N)[clusters[1][c]])
        
    
    elif dataset_N_aux.shape[0]==0:
        clusters_N = [] 
        templates = []
        frames_new = [] 
    else: 
        clusters = (1, [[0]])
        templates_N = dataset_N_aux.copy()
        clusters_N = [] 
        templates = []  
        frames_new = ()
        dt = 1/frequency
        for c in range(clusters[0]):
            dy = np.diff(templates_N[:,c])/dt
            dy = np.concatenate(([0], dy)) 
            i_peaks = scipy.signal.find_peaks(-templates_N[:,c], width = 2.5)[0] #lavorare un po' qua
            der = dy[i_peaks] 
            if len(scipy.signal.find_peaks(-templates_N[:,c], prominence=4)[0])==1 and len(clusters[1][c])>2 and len(scipy.signal.find_peaks(templates_N[:,c], prominence=25)[0])<=1: 
                clusters_N.append(clusters[1][c])
                templates.append(templates_N[:, c])
                frames_new = set(frames_new)|set(np.array(frames_N)[clusters[1][c]])


    templates_N = np.array(templates).T

    clusters = [len(clusters_N), clusters_N] 

    for var in list(locals()):
        if var != 'cluster' or var != 'templates_N' or var != 'frames_N':
            del locals()[var]
    import gc
    gc.collect()


    return clusters, templates_N, frames_N #list(frames_new) #modificare in modo da avere solo i frames degli spikes che tengo

def TemplateMatching(data, templates, thresh = 0.95):
    """
    Templates matching
    
    Args:
        data (np.ndarray): vector representing the signal where compute the templates matching.
        templates (np.ndarray): matrix number of templates x number of frames representing found templates.
        thresh (float): match when Pearson's correlation coefficient upper than the threshold. Defaults to 0.95.
    
    Returns:
        frames (list): frames idx associated to the matching template.
        data (np.ndarray): signal after matching subtraction.
        dataset (list): matching frames waveforms.
        dataset_idx (list): association between dataset element and matching templates.
    """

    n = templates.shape[0]
    if n==0:
        return [], data, [], []
    else:
        size = int(templates.shape[1]/2)
        x = set(np.arange(data.shape[0]))
        x_1 = set(np.arange(size))  
        x_2 = set(np.arange(size+4)+data.shape[0]-size-4) #5 generico 
        y = np.array(sorted(x-x_1-x_2))
        frames = []
        dataset_idx = [] 
        for c in range(n):
            frames.append([])
            dataset_idx.append([])
        dataset = []
        '''Spikes detection by templates matching'''
        i = y[0]   
        while i <= y[-1] :
            #t=time.time()
            corr = [] 
            for c in range(n):
                corr.append(pearsonr(templates[c]/np.linalg.norm(templates[c]), data[i-20:i+21]/np.linalg.norm(data[i-20:i+21]))[0])
            corr = np.array(corr)
            if np.max(corr)>thresh:
                idx = np.where(corr==np.max(corr))[0][0]
                corr_aux = np.zeros(5) 
                for j in range(5):
                    corr_aux[j]= pearsonr(templates[idx]/np.linalg.norm(templates[idx]), data[i+j-20:i+j+21]/np.linalg.norm(data[i+j-20:i+j+21]))[0]
                j_max = np.where(corr_aux==np.max(corr_aux))[0][0]
                frames[idx].append(i+j_max)
                dataset_idx[idx].append(len(dataset))
                dataset.append(templates[idx]/np.linalg.norm(templates[idx])*np.linalg.norm(data[i+j_max-20:i+j_max+21]))
                data[i+j_max-20:i+j_max+21] = data[i+j_max-20:i+j_max+21]-templates[idx]/np.linalg.norm(templates[idx])*np.linalg.norm(data[i+j_max-20:i+j_max+21])
            else:
                i = i+1
            #print(str(i)+': '+str(time.time()-t)) 
        
        for var in list(locals()):
            if var != 'frames' or var != 'data' or var != 'dataset' or var !='dataset_idx':
                del locals()[var]
        import gc
        gc.collect()

        return frames, data, dataset, dataset_idx
    
def CrossCorrelogram(f_cluster_1, f_cluster_2, SamplingRate, NumFrames):
    """
    _summary_
    
    Args:
        f_cluster_1 (_type_): _description_.
        f_cluster_2 (_type_): _description_.
        SamplingRate (_type_): _description_.
        NumFrames (_type_): _description_.
    """
    spike_times_1 = f_cluster_1/SamplingRate
    spike_times_2 = f_cluster_2/SamplingRate

    t_stop = NumFrames/SamplingRate
    st_A = SpikeTrain(spike_times_1 * pq.s, t_stop=t_stop * pq.s)
    st_B = SpikeTrain(spike_times_1 * pq.s, t_stop=t_stop * pq.s)
    ccg, bins = cross_correlation_histogram(st_A, st_B, window=41/SamplingRate* pq.s, bin_size=1/SamplingRate*1000*pq.ms, border_correction=False)

    ccg_counts = ccg.magnitude.flatten()

    # Indici relativi ai bin centrali (±1 ms = ±10 bin)
    center_bin = len(ccg_counts) // 2
    center_region = ccg_counts[center_bin - int(SamplingRate/1000) : center_bin + int(SamplingRate/1000)+1]

    # Zona laterale (±3–5 ms)
    side_region = np.concatenate([
        ccg_counts[:int(SamplingRate/1000)],         # bin -5 ms a -4 ms
        ccg_counts[-int(SamplingRate/1000):]         # bin +4 a +5 ms
    ])

    mean_center = np.mean(center_region)
    mean_side = np.mean(side_region)

    print("Conteggio medio zona centrale:", mean_center)
    print("Conteggio medio zona laterale:", mean_side)

    if mean_center < 0.25 * mean_side:
        print("→ CCG refrattario: possibile overclustering")
    else:
        print("→ Nessuna chiara refrattarietà")

def Wrapper(args_tuple):
    """
    _summary_
    
    Args:
        args_tuple (_type_): _description_.
    
    Returns:
        _type_: _description_.
    """
    final_templates, final_frames = SacchiSpikesSorting(*args_tuple)

    for var in list(locals()):
        if var != 'final_templates' or var != 'final_frames':
            del locals()[var]
    import gc
    gc.collect()

    return final_templates[0], final_frames[0]

def stima_numero_processi(ram_per_process_gb=2.0, ram_riservata_gb=6.0):
    ram_disponibile = psutil.virtual_memory().available / (1024 ** 3)  # da byte a GB
    processi_max = int((ram_disponibile - ram_riservata_gb) // ram_per_process_gb)

    for var in list(locals()):
        if var != 'processi_max':
            del locals()[var]
    import gc
    gc.collect()
    return max(1, processi_max)

def ChannelSpksort(ch):
    """
    _summary_
    
    Args:
        ch (_type_): _description_.
    
    Returns:
        _type_: _description_.
    """
    row = ch//64
    col = ch % 64
    rows = np.arange(row-2,row+2+1)
    cols = np.arange(col-2, col+2+1)
    rows = rows[rows>=0]
    cols = cols[cols>=0]
    chs = []
    for i in rows:
        for j in cols:
            chs.append(i*64+j)
    chs = np.array(sorted(chs))
    idx_ch = np.where(chs==ch)[0]

    for var in list(locals()):
        if var != 'chs' or var != 'idx_ch':
            del locals()[var]
    import gc
    gc.collect()

    return chs, idx_ch

def SacchiSpikesSorting(Data_ch, SamplingRate, algo = 'Leiden', chs = [0], channel_of_study = 0, notchcut=50, lowcut=300, highcut=3000, distance = 'rho', method_HC = 'complete', criterion_HC = 'distance', method_KM = 'silhouette', max_iter_FCM=10, threshold_variance = 0.9, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, fuzzy_parameter = 1, noise = 0, threshold_dendrogram = 0.33, max_classes = [2], threshold_Leiden = 0.9, p_minkowski = 2, normalization = 'OFF', norm_mode = 'min_max_single'):
    """
    Spikes sorting pipeline
    
    Args:
        brw_filename (str): name of the file (with the path) and its extension .brw.
        wellID (str): well ID.
        algo (str): clustering algo. Defaults to 'Leiden'.
        chs (int): list of channels ID on which comppute spikes sorting. Defaults to [0:4096].
        notchcut (int): frequency for Notch filter. Defaults to 50.
        lowcut (int): lower frequency for band pass filter. Defaults to 300.
        highcut (int): upper frequency for band pass filter. Defaults to 3000.
        distance (str): metric to compute distances in clustering. Defaults to 'rho'.
        method_HC (str): linkage method. Defaults to 'complete'.
        criterion_HC (str): hierarchical clustering criterion. Defaults to 'distance'.
        method_KM (str): method to compute the optimal number of centroids in KM and FCM or relatives. Defaults to 'silhouette'.
        max_iter_FCM (int): maximum number of iterations in FCM. Defaults to 10.
        threshold_variance (float): explained variance after PCA. Defaults to 0.9.
        w_max (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilon_EDR (float): EDR threshold. Defaults to 0.001.
        epsilon_LCSS (float): LCSS threshold. Defaults to 0.001.
        fuzzy_parameter (int): FCM parameter. Defaults to 1.
        noise (int): amount of noise in percentage to add to data. Defaults to 0.
        threshold_dendrogram (float): cut height of the dendrogram. Defaults to 0.33.
        max_classes (int): maximum number of classes for clustering. Defaults to 1.
        threshold_Leiden (float): Leiden threshold. Defaults to 0.9.
        p_minkowski (int): Minkowski parameter. Defaults to 2.
        frequency (int): STS parameter. Defaults to 1000.
        normalization (str): To applying normalization. Defaults to 'OFF'.
        norm_mode (str): If normalization applied, to select the modality. Defaults to 'min_max_single'.
    
    Returns:
        final_templates (list): vector representing neurons shapes.
    """

    '''READING'''
    # t=time.time()
    #brw = brw_f.ReadBRW(brw_filename, wellID)
    #SamplingRate = float(brw.attrs['SamplingRate'])
    #toc = np.array(brw['TOC'])
    #NumFrames = toc[toc.shape[0]-1,1]

    '''
    t = time.time()
    DataFull_1, frames_index_1 = brw_f.ReadingRawData(brw, wellID, SamplingRate, 0, 30)
    DataFull_2, frames_index_2 = brw_f.ReadingRawData(brw, wellID, SamplingRate, 30, 30)
    DataFull_3, frames_index_3 = brw_f.ReadingRawData(brw, wellID, SamplingRate, 60, 30)
    DataFull_4, frames_index_4 = brw_f.ReadingRawData(brw, wellID, SamplingRate, 90, 30)
    DataFull_5, frames_index_5 = brw_f.ReadingRawData(brw, wellID, SamplingRate, 120, 30)
    DataFull_6, frames_index_6 = brw_f.ReadingRawData(brw, wellID, SamplingRate, 150, NumFrames/SamplingRate-150)
    f_1 = len(frames_index_1)
    f_2 = len(frames_index_2)
    f_3 = len(frames_index_3)
    f_4 = len(frames_index_4)
    f_5 = len(frames_index_5)
    f_6 = len(frames_index_6)
    NumFrames = f_1 + f_2 + f_3 + f_4 + f_5 + f_6
    DataFull = np.zeros((NumFrames, 4096))
    DataFull[0:f_1] = DataFull_1.copy()
    DataFull[f_1:f_1+f_2] = DataFull_2.copy()
    DataFull[f_1+f_2:f_2+f_3] = DataFull_3.copy()
    DataFull[f_2+f_3:f_3+f_4] = DataFull_4.copy()
    DataFull[f_3+f_4:f_4+f_5] = DataFull_5.copy()
    DataFull[f_4+f_5:NumFrames] = DataFull_6.copy()
    median_dato = np.median(DataFull,1)
    print('reading: '+str(time.time()-t))
    '''
    #DataFull_2m, frames_index_2m = brw_f.ReadingRawData(brw, wellID, SamplingRate, 10, 10) 
    #DataFull_2m = np.fromfile('C:/Users/BioCAM User/Desktop/LORENZO/KILOSORT4/3D_BrAIn_float32_40964ch_10s.bin', dtype=np.float32)
    #DataFull_2m = DataFull_2m.reshape(4096, int(DataFull_2m.shape[0]/4096)).T
    DataFull = Data_ch.copy()
    median_dato = np.median(DataFull,1)
    # print('Data reading: '+str(time.time()-t))
    #''' 
    NumFrames = DataFull.shape[0]
    final_templates_tot = []
    frames_final_tot = []
    for ch in chs:
        tempo = time.time()
        if np.max(np.abs(DataFull[:,ch]))!=8000:
            '''LEARNING TEMPLATES''' 
            '''25 Channels templates''' #a neuron fires to a distance between 50-100 um and the electrodes have a distance from each other of 42 um; do we consider 25 electrodes? (the center and the 24 around)

            # t = time.time()
            templates = [] 
            templates_neg = []
            templates_pos = []
            for i in range(Data_ch.shape[1]):
                channel = i
                DataFull[:, channel] = DataFull[:, channel] - median_dato 
                DataFull[:, channel] = brw_f.Notch_filter(DataFull[:, channel], notchcut, SamplingRate)  
                DataFull[:, channel] = brw_f.bandpass_filter(DataFull[:, channel], lowcut, highcut, SamplingRate) 
                DataFull[:, channel] = DataFull[:, channel] - np.mean(DataFull[:, channel])
                if np.max(np.abs(DataFull[:,channel]))!=8000:
                    clusters_P, templates_P, frames_P = TemplateNeg(data=-DataFull, ch=channel, parameter=4.5, algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, p_minkowski=p_minkowski, frequency=SamplingRate)
                    clusters_N, templates_N, frames_N = TemplateNeg(data=DataFull, ch=channel, parameter=4.5, algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, p_minkowski=p_minkowski, frequency=SamplingRate)
                    if channel == ch:
                        frames_P_thresholding=frames_P
                        frames_N_thresholding=frames_N
                    if clusters_N[0]+clusters_P[0]>0:
                        if clusters_N[0]>0:
                            for c in range(templates_N.shape[1]):
                                templates.append(templates_N[:,c])
                                templates_neg.append(templates_N[:,c])
                        if clusters_P[0]>0:
                            for c in range(templates_P.shape[1]):
                                templates.append(-templates_P[:,c])
                                templates_pos.append(-templates_P[:,c])

            templates = np.array(templates)
            '''
            plt.figure()
            for i in range(templates.shape[0]):
                plt.plot(np.arange(41), templates[i])
            plt.savefig('templates')
            # '''

            if templates.shape[0]>1:
                df = pd.DataFrame(templates.T)
                columns = FindCorrelation(df, thresh = 0.95)
                templates = templates[columns,:]  
            '''
            plt.figure()
            for i in range(templates.shape[0]):
                plt.plot(np.arange(41), templates[i])
            plt.savefig('templates_new')
            # ''' 
            # print('Templates learning: '+str(time.time()-t))

            '''TEMPLATES MATCHING'''
            # t = time.time()
            data=DataFull[:,ch].copy() 
            frames, data_noise, dataset_list, dataset_idx = TemplateMatching(data, templates, thresh = 0.95) 

            signals=[] 
            dataset = [] 
            frames_tot =[] 
            dataset_idx_tot = []
            for c in range(templates.shape[0]):
                signals.append(templates[c])
                frames_tot.append(frames[c])
                dataset_idx_tot.append(dataset_idx[c])

            for k in range(len(dataset_list)):
                dataset.append(dataset_list[k])
                
            par = [4.5, 4, 3.5, 3, 4.5, 4] #guardare cosa succede aggiungendo gli ultimi 2
            p = 0
            DataFull[:,ch]=data_noise.copy()
            while p<len(par):
                clusters_P, templates_P, frames_P = TemplateNeg(data=-DataFull, ch=ch, parameter=par[p], algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, p_minkowski=p_minkowski, frequency=SamplingRate)
                clusters_N, templates_N, frames_N = TemplateNeg(data=-DataFull, ch=ch, parameter=par[p], algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, p_minkowski=p_minkowski, frequency=SamplingRate)
                if clusters_N[0]+clusters_P[0]>0:
                    l=len(signals)
                    templates_aux = []
                    if clusters_N[0]>0:
                        for c in range(templates_N.shape[1]):
                            signals.append(templates_N[:,c])
                            templates_aux.append(templates_N[:,c])
                            templates_neg.append(templates_N[:,c])
                    if clusters_P[0]>0:
                        for c in range(templates_P.shape[1]):
                            signals.append(-templates_P[:,c])
                            templates_aux.append(-templates_P[:,c]) 
                            templates_pos.append(-templates_P[:,c])
                    templates_aux = np.array(templates_aux)
                    frames, data_noise, dataset_list, dataset_idx = TemplateMatching(DataFull[:, ch], templates_aux, thresh = 0.95)  
                    for c in range(len(frames)):
                        frames_tot.append(frames[c])
                        dataset_idx_tot.append(list(np.array(dataset_idx[c])+len(dataset)))
                    for k in range(len(dataset_list)):
                        dataset.append(dataset_list[k])
                    DataFull[:,ch]=data_noise
                        
                else:
                    for i in np.array(sorted(set(frames_N)|set(frames_P))):
                        if i>=20 and i<NumFrames-20:
                            if i in np.array(sorted(set(frames_N))) and len(templates_neg)>0:
                                corr = []
                                for c in range(len(signals)):
                                    corr.append(pearsonr(signals[c]/np.linalg.norm(signals[c]), DataFull[i-20:i+21, ch]/np.linalg.norm(DataFull[i-20:i+21, ch]))[0])
                                corr = np.array(corr)
                                if np.max(corr)>0.5:
                                    idx = np.where(corr==np.max(corr))[0][0]
                                    frames_tot[idx].append(i)
                                    dataset_idx_tot[idx].append(len(dataset)) 
                                    dataset.append(signals[idx]/np.linalg.norm(signals[idx])*np.linalg.norm(DataFull[i-20:i+21, ch]))
                                    DataFull[i-20:i+21, ch] = DataFull[i-20:i+21, ch]-signals[idx]/np.linalg.norm(signals[idx])*np.linalg.norm(DataFull[i-20:i+21, ch])
                            if i in np.array(sorted(set(frames_P))) and len(templates_pos)>0:
                                corr = [] 
                                for c in range(len(signals)):
                                    corr.append(pearsonr(signals[c]/np.linalg.norm(signals[c]), DataFull[i-20:i+21, ch]/np.linalg.norm(DataFull[i-20:i+21, ch]))[0])
                                corr = np.array(corr)
                                if np.max(corr)>0.5:
                                    idx = np.where(corr==np.max(corr))[0][0]
                                    frames_tot[idx].append(i) 
                                    dataset_idx_tot[idx].append(len(dataset))
                                    dataset.append(signals[idx]/np.linalg.norm(signals[idx])*np.linalg.norm(DataFull[i-20:i+21, ch]))
                                    DataFull[i-20:i+21, ch] = DataFull[i-20:i+21, ch]-signals[idx]/np.linalg.norm(signals[idx])*np.linalg.norm(DataFull[i-20:i+21, ch])
                            
                    p=p+1

            # print('Templates matching: '+str(time.time()-t))
            
            n_tot = len(dataset)
            D = np.zeros((n_tot, 41)) 
            for k in range(n_tot):
                D[k] = dataset[k] 

            '''Graph-based clustering'''
            # t = time.time()
            if D.shape[0]>0:
                clusters_L, G, partition = stratification_sktime.Leiden_Algo(D.T, threshold_Leiden=0.95, distance='rho', p_minkowski=p_minkowski, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, SamplingRate=SamplingRate) #modificare poi inerendo clustering...
                n_classes = len(clusters_L)
            else:
                n_classes = 0
            clusters = []
            for i in range(n_classes):
                if len(clusters_L[i][0])>0:
                    clusters.append(clusters_L[i][0])
            n_classes = len(clusters)
            # print('Leiden algorithm: '+str(time.time()-t))

            final_templates = np.zeros((len(clusters), 41))
            #legend_graph=[]
            frames_final = []
            #plt.figure()
            #plt.title('Neurons shapes candidate')
            for c in range(len(clusters)):
                final_templates[c] = np.mean(D[clusters[c]].T,1).reshape(41)
                frames_final.append([])
                #legend_graph.append(mpatches.Patch(color= colors[c], label='Neuron '+str(c+1)))
                #plt.plot(np.arange(41), np.mean(D[clusters[c]].T,1).reshape(41), color=colors[c], label='Neuron '+str(c+1))
            #plt.legend()
            #plt.ylabel('(uV)')
            #plt.xlabel('41 frames ≃ 2 ms')
            #plt.savefig('final_templates')

            #G.vs["size"] = [2 + deg/20 for deg in G.degree()]
            #G.vs["color"] = v_colors
            #ig.plot(G, layout=G.layout("fr"), vertex_size=G.vs['size'], vertex_color=G.vs['color'], edge_color='lightgray', target='Leiden_graph.png')
            #img = plt.imread("Leiden_graph.png")
            #plt.figure(figsize=(8, 8))
            #plt.imshow(img)
            #plt.axis("on")
            #plt.title("Leiden graph", fontsize=16)
            #plt.legend(handles=legend_graph, loc='lower right')
            #plt.show()
            #plt.savefig('Leiden_graph')
            

            '''Merging Tree'''
            if len(clusters)>1:
                # t = time.time()
                merge.merging_tree(G,partition)
                # print('Merging Tree: '+str(time.time()-t))


            '''Reconstruction''' #rivedere
            reconstruction = np.zeros(NumFrames)
            '''
            DataFull[0:f_1m, ch] = DataFull_1m[:, ch].copy()
            DataFull[f_1m:f_1m+f_2m, ch] = DataFull_2m[:, ch].copy()
            DataFull[f_2m+f_1m:NumFrames, ch] = DataFull_3m[:, ch].copy()
            ''' 
            #DataFull[:, ch] = DataFull[:, ch] - median_dato
            #DataFull[:, ch] = brw_f.Notch_filter(DataFull[:, ch], notchcut, SamplingRate)  
            #DataFull[:, ch] = brw_f.bandpass_filter(DataFull[:, ch], lowcut, highcut, SamplingRate)
            #DataFull[:, channel] = DataFull[:, ch] - np.mean(DataFull[:, ch])
            for c in range(len(frames_tot)):
                for i in range(len(frames_tot[c])):
                    f = frames_tot[c][i]
                    aux_idx = dataset_idx_tot[c][i]
                    high_peak = dataset[aux_idx][20]
                    j=0
                    while j < (len(final_templates)):
                        if aux_idx in clusters[j]:
                            reconstruction[f-20:f+21]+=final_templates[j]/final_templates[j][20]*high_peak
                            frames_final[j].append(f)
                            j=(len(final_templates))
                        else:
                            j=j+1
            DataFull[:,ch] = Data_ch[:,ch].copy()
        else:
            final_templates = []
            reconstruction = np.zeros(NumFrames)
            frames_final = []

        final_templates_tot.append(final_templates)
        frames_final_tot.append(frames_final)
        print('SPIKES SORTING ELETTRODO ' +str(channel_of_study)+': '+str(time.time()-tempo))
        print('---')
    
    for var in list(locals()):
        if var != 'final_templates_tot' or var != 'frames_final_tot':
            del locals()[var]
    import gc
    gc.collect()

    return final_templates_tot, frames_final_tot

def LinkChsSpksort(results):
    """
    _summary_
    
    Args:
        results (_type_): _description_.
    
    Returns:
        _type_: _description_.
    """
    n_chs = len(results)
    common_neuron = np.zeros((n_chs, n_chs))
    for i in range(n_chs):
        for j in np.arange(i+1, n_chs):
            for s in range(len(results[i][0])):
                for k in range(len(results[j][0])):
                    if  pearsonr(results[i][0][s], results[j][0][k])[0]>=0.95:
                        common_neuron[i][j]+=1
    
    for var in list(locals()):
        if var != 'common_neuron':
            del locals()[var]
    import gc
    gc.collect()

    return common_neuron



