"""Spike sorting and detection functions for neural data analysis."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pywt
import math
import scipy
import time
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import pearsonr
from statistics import median
import igraph as ig
from igraph import Graph, plot
import leidenalg as la
import random
from . import brw_functions as brw_f
from . import stratification
from . import merging_tree as merge
import matplotlib.patches as mpatches
try:
    from neo import SpikeTrain 
    import quantities as pq
    from elephant.spike_train_correlation import cross_correlation_histogram
except ImportError:
    # Optional dependencies for correlation analysis
    SpikeTrain = None
    pq = None
    cross_correlation_histogram = None
from concurrent.futures import ProcessPoolExecutor
import psutil
import os
import gc


def FindCorrelation(df, thresh=0.9, verbose=False):
    """Find and remove highly correlated features from dataframe.
    
    Uses correlation analysis to identify redundant variables and returns
    a list of representative variables, keeping one from each correlated group.

    Args:
        df (pd.DataFrame): input dataframe with features as columns
        thresh (float, optional): correlation threshold above which features are considered redundant. Defaults to 0.9.
        verbose (bool, optional): if True, print progress information. Defaults to False.

    Raises:
        ValueError: if dataframe has only one variable

    Returns:
        list: indices of non-redundant features to keep
    """    
    corrMatrix = df.corr()
    varnum = corrMatrix.shape[0]
    
    if varnum == 1:
        raise ValueError("only one variable given")
    
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
    temp_matrix = corrMatrix.copy()

    delete_col = list(original_order)
    original_order = list(original_order)
    new_order = list(new_order)
    col = []
    
    while np.any(temp_matrix[~np.isnan(temp_matrix)] > thresh) and len(new_order) > 0:
        if verbose:
            print("All correlations <=", thresh)
            break
        idx = np.where(np.array(temp_matrix[new_order[0]]) > thresh)[0]
        for i in range(len(idx)):
            delete_col.remove(original_order[idx[i]])
            new_order.remove(original_order[idx[i]])
        col.append(new_order[0])
        delete_col.remove(new_order[0])
        new_order.remove(new_order[0])
        original_order = delete_col.copy()
        temp_matrix = temp_matrix[new_order]
        temp_matrix = temp_matrix.loc[delete_col]

    if temp_matrix.shape[0] > 0:
        for i in range(temp_matrix.shape[0]):
            col.append(temp_matrix.columns[i])

    return sorted(col)


def SpikesDetection(data, step, threshold, aux_spike):
    """Detect spikes on negative or positive peaks using threshold criteria.
    
    Detects spikes where the signal crosses a threshold defined as t = mu ± threshold*sigma,
    where mu is the mean and sigma is the standard deviation of signal segments.

    Args:
        data (array): 1D signal array
        step (int): window size for computing mean and std
        threshold (float): threshold factor in units of standard deviations
        aux_spike (str): "pos" for positive spikes, "neg" for negative spikes

    Returns:
        list: frame indices where spikes are detected
    """    
    data = data[:data.shape[0] - (data.shape[0] % step)]
    data_reshaped = data.reshape(-1, step)
    mu = np.mean(data_reshaped, axis=1)
    sigma = np.std(data_reshaped, axis=1)
    data_reshaped_aux = data_reshaped - mu[:, np.newaxis]
    th_sigma = threshold * sigma
    
    frames = []
    for ii in range(data_reshaped_aux.shape[0]):
        row = data_reshaped_aux[ii, :]
        if aux_spike == "pos":
            peaks, properties = find_peaks(row, height=th_sigma[ii])
        else:
            peaks, properties = find_peaks(-row, height=th_sigma[ii])
        if len(peaks) > 0:
            peaks = peaks[np.argmax(properties["peak_heights"])]
            peaks = peaks + step * ii
            frames.append(peaks)

    return frames


def TemplateNeg(data, ch, parameter=4.5, algo='Leiden', distance='rho', method_HC='complete', 
                criterion_HC='distance', method_KM='silhouette', max_iter_FCM=10, 
                threshold_variance=0.9, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, 
                fuzzy_parameter=1, noise=0, threshold_dendrogram=0.33, max_classes=[2], 
                threshold_Leiden=0.9, p_minkowski=2, frequency=1000, normalization='OFF', 
                norm_mode='min_max_single'):
    """Learn spike templates for negative (and positive) peaks using clustering.
    
    Detects spikes and learns representative templates through clustering of spike waveforms.
    Removes low-quality detections and correlated templates to obtain clean spike classes.

    Args:
        data (array): 2D data array [samples, channels]
        ch (int): channel index for template learning
        parameter (float, optional): spike detection threshold in std units. Defaults to 4.5.
        algo (str, optional): clustering algorithm. Defaults to 'Leiden'.
        distance (str, optional): distance metric. Defaults to 'rho'.
        method_HC, criterion_HC, method_KM: clustering parameters
        frequency (int, optional): sampling rate in Hz. Defaults to 1000.
        Other parameters: clustering-specific parameters

    Returns:
        tuple: (clusters, templates, frames) where:
            - clusters (list): cluster indices
            - templates (array): learned spike templates
            - frames (list): detected spike frame indices
    """
    DataChannel = data[:, ch].copy()
    NumFrames = DataChannel.shape[0]
    step = int(frequency * 0.01)
    
    frames_N = set()
    t = 0
    while t < NumFrames - step:
        frames_neg = SpikesDetection(data[t:t+step, :], ch, parameter, "neg") + t
        t = t + step
        frames_N = frames_N | set(frames_neg)
    
    frames_neg = SpikesDetection(data[t:NumFrames, :], ch, parameter, "neg") + t
    frames_N = frames_N | set(frames_neg)
    frames_N = sorted(frames_N)

    # PROVA PER MIGLIORARE MA DA RICONTROLLARE
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

    # Extract waveforms around detected spikes
    dataset_N = np.zeros((len(frames_N), 41))
    for k in range(len(frames_N)):
        peak_frame = frames_N[k]
        if peak_frame < 20:
            dataset_N[k, 20-peak_frame:41] = DataChannel[0:peak_frame+21]
        elif peak_frame >= NumFrames - 20:
            dataset_N[k, 0:NumFrames-peak_frame+20] = DataChannel[peak_frame-20:NumFrames]
        else:
            dataset_N[k] = DataChannel[peak_frame-20:peak_frame+21]
    
    dataset_N_aux = dataset_N.copy()
    
    if dataset_N_aux.shape[0] > 1:
        clusters = stratification.RecursiveClustering(data=dataset_N_aux, algo=algo, distance=distance, 
                                                     method_HC=method_HC, criterion_HC=criterion_HC, 
                                                     method_KM=method_KM, max_iter_FCM=max_iter_FCM, 
                                                     threshold_variance=threshold_variance, w_max=w_max, 
                                                     g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, 
                                                     fuzzy_parameter=fuzzy_parameter, noise=noise, 
                                                     threshold_dendrogram=threshold_dendrogram, 
                                                     max_classes=max_classes, threshold_Leiden=threshold_Leiden, 
                                                     SamplingRate=frequency, p_minkowski=p_minkowski, 
                                                     normalization=normalization, norm_mode=norm_mode)

        templates_N = []
        for c in range(clusters[0]):
            cluster_data = dataset_N[clusters[1][c]]
            cluster_data = cluster_data.reshape((cluster_data.shape[-2], cluster_data.shape[-1]))
            mu = np.mean(cluster_data, 0)
            templates_N.append(mu)
        templates_N = np.array(templates_N).T
        
        # Remove correlated templates
        df = pd.DataFrame(templates_N)
        corr = np.array(df.corr()) - np.eye(templates_N.shape[1])
        idxs = set(np.arange(templates_N.shape[1]))
        
        while np.max(corr) >= 0.95:
            idxs_del = np.where(corr == np.max(corr))
            a, b = idxs_del[0][0], idxs_del[1][0]
            idxs = idxs - {a, b}
            clusters_new = [clusters[1][i] for i in idxs]
            clusters_new.append(list(set(clusters[1][a]) | set(clusters[1][b])))
            clusters = (len(clusters_new), clusters_new)
            
            templates_N = []
            for c in range(clusters[0]):
                cluster_data = dataset_N[clusters[1][c]]
                cluster_data = cluster_data.reshape((cluster_data.shape[-2], cluster_data.shape[-1]))
                mu = np.mean(cluster_data, 0)
                templates_N.append(mu)
            templates_N = np.array(templates_N).T
            df = pd.DataFrame(templates_N)
            corr = np.array(df.corr()) - np.eye(templates_N.shape[1])
            idxs = set(np.arange(templates_N.shape[1]))

        # Quality filtering
        clusters_N = []
        templates = []
        dt = 1 / frequency
        
        for c in range(clusters[0]):
            dy = np.diff(templates_N[:, c]) / dt
            dy = np.concatenate(([0], dy))
            
            if (len(scipy.signal.find_peaks(-templates_N[:, c], prominence=4)[0]) == 1 and 
                len(clusters[1][c]) > 2 and 
                len(scipy.signal.find_peaks(templates_N[:, c], prominence=25)[0]) <= 1):
                clusters_N.append(clusters[1][c])
                templates.append(templates_N[:, c])

    elif dataset_N_aux.shape[0] == 1:
        clusters_N = []
        templates = []
    else:
        clusters_N = []
        templates = []

    templates_N = np.array(templates).T if len(templates) > 0 else np.array([])
    clusters = [len(clusters_N), clusters_N]

    gc.collect()
    return clusters, templates_N, frames_N


def TemplateMatching(data, templates, thresh=0.95):
    """Match signal to learned spike templates using correlation.
    
    Finds occurrences of learned spike templates in signal using template matching
    with Pearson correlation coefficient as similarity measure.

    Args:
        data (array): 1D signal where to search for templates
        templates (array): 2D array of learned templates (time_points, n_templates)
        thresh (float, optional): correlation threshold for template matching. Defaults to 0.95.

    Returns:
        tuple: (frames, data_residual, dataset, dataset_idx) where:
            - frames (list): frame indices of matched spikes for each template
            - data_residual (array): signal after subtracting matched templates
            - dataset (list): matched spike waveforms
            - dataset_idx (list): mapping between matches and templates
    """
    n = templates.shape[0]
    if n == 0:
        return [], data.copy(), [], []
    
    size = int(templates.shape[1] / 2)
    x = set(np.arange(data.shape[0]))
    x_1 = set(np.arange(size))
    x_2 = set(np.arange(size + 4) + data.shape[0] - size - 4)
    y = np.array(sorted(x - x_1 - x_2))
    
    frames = [[] for _ in range(n)]
    dataset_idx = [[] for _ in range(n)]
    dataset = []
    data_copy = data.copy()
    
    i = y[0]
    while i <= y[-1]:
        corr = []
        for c in range(n):
            try:
                r = pearsonr(templates[c] / np.linalg.norm(templates[c]), 
                            data_copy[i-20:i+21] / np.linalg.norm(data_copy[i-20:i+21]))[0]
            except:
                r = 0
            corr.append(r)
        
        corr = np.array(corr)
        if np.max(corr) > thresh:
            idx = np.argmax(corr)
            # Refine peak location
            corr_aux = np.zeros(5)
            for j in range(5):
                try:
                    corr_aux[j] = pearsonr(templates[idx] / np.linalg.norm(templates[idx]),
                                          data_copy[i+j-20:i+j+21] / np.linalg.norm(data_copy[i+j-20:i+j+21]))[0]
                except:
                    corr_aux[j] = 0
            
            j_max = np.argmax(corr_aux)
            frames[idx].append(i + j_max)
            dataset_idx[idx].append(len(dataset))
            template_norm = templates[idx] / np.linalg.norm(templates[idx])
            waveform = data_copy[i+j_max-20:i+j_max+21]
            dataset.append(template_norm * np.linalg.norm(waveform))
            data_copy[i+j_max-20:i+j_max+21] -= template_norm * np.linalg.norm(waveform)
        
        i += 1
    
    gc.collect()
    return frames, data_copy, dataset, dataset_idx


def CrossCorrelogram(f_cluster_1, f_cluster_2, SamplingRate, NumFrames):
    """Compute cross-correlogram between two spike trains.
    
    Calculates cross-correlation histogram to detect synchrony between neurons,
    useful for identifying potential cell pairs and detecting refractory violations.

    Args:
        f_cluster_1 (array): spike frame indices for first cluster
        f_cluster_2 (array): spike frame indices for second cluster
        SamplingRate (float): sampling rate in Hz
        NumFrames (int): total number of frames in recording

    Returns:
        None (prints analysis results)
    """
    spike_times_1 = f_cluster_1 / SamplingRate
    spike_times_2 = f_cluster_2 / SamplingRate
    t_stop = NumFrames / SamplingRate
    
    st_A = SpikeTrain(spike_times_1 * pq.s, t_stop=t_stop * pq.s)
    st_B = SpikeTrain(spike_times_2 * pq.s, t_stop=t_stop * pq.s)
    ccg, bins = cross_correlation_histogram(st_A, st_B, window=41/SamplingRate*pq.s, 
                                           bin_size=1/SamplingRate*1000*pq.ms, border_correction=False)

    ccg_counts = ccg.magnitude.flatten()
    center_bin = len(ccg_counts) // 2
    center_region = ccg_counts[center_bin - int(SamplingRate/1000):center_bin + int(SamplingRate/1000)+1]
    side_region = np.concatenate([ccg_counts[:int(SamplingRate/1000)], ccg_counts[-int(SamplingRate/1000):]])

    mean_center = np.mean(center_region)
    mean_side = np.mean(side_region)

    print("Mean counts central zone:", mean_center)
    print("Mean counts lateral zone:", mean_side)
    if mean_center < 0.25 * mean_side:
        print("→ Refractory CCG: possible overclustering")
    else:
        print("→ No clear refractoriness")


def EstimateNumProcesses(ram_per_process_gb=2.0, ram_riservata_gb=6.0):
    """Estimate optimal number of parallel processes based on available RAM.
    
    Args:
        ram_per_process_gb (float, optional): estimated RAM per process in GB. Defaults to 2.0.
        ram_riservata_gb (float, optional): reserved RAM not to be used. Defaults to 6.0.

    Returns:
        int: recommended number of processes
    """
    ram_disponibile = psutil.virtual_memory().available / (1024 ** 3)
    processi_max = int((ram_disponibile - ram_riservata_gb) // ram_per_process_gb)
    gc.collect()
    return max(1, processi_max)


def ChannelSpksort(ch):
    """Get neighboring channels for spike sorting (5x5 neighborhood).
    
    Args:
        ch (int): channel index (0-4095 for 64x64 grid)

    Returns:
        tuple: (neighboring_channels, index_of_ch_in_neighbors)
    """
    row = ch // 64
    col = ch % 64
    rows = np.arange(row-2, row+3)
    cols = np.arange(col-2, col+3)
    rows = rows[rows >= 0]
    cols = cols[cols < 64]
    
    chs = []
    for i in rows:
        for j in cols:
            chs.append(i * 64 + j)
    chs = np.array(sorted(chs))
    idx_ch = np.where(chs == ch)[0]
    gc.collect()
    return chs, idx_ch


def SacchiSpikeSorting(Data_ch, SamplingRate, algo='Leiden', chs=[0], channel_of_study=0, 
                      notchcut=50, lowcut=300, highcut=3000, distance='rho', 
                      method_HC='complete', criterion_HC='distance', method_KM='silhouette', 
                      max_iter_FCM=10, threshold_variance=0.9, w_max=1, g=1, 
                      epsilon_EDR=0.001, epsilon_LCSS=0.001, fuzzy_parameter=1, noise=0, 
                      threshold_dendrogram=0.33, max_classes=[2], threshold_Leiden=0.9, 
                      p_minkowski=2, normalization='OFF', norm_mode='min_max_single'):
    """Complete spike sorting pipeline with template learning and matching.
    
    Performs unsupervised spike sorting including: spike detection, template learning,
    template matching, and graph-based clustering to identify distinct neurons.

    Args:
        Data_ch (array): 2D data array [samples, channels]
        SamplingRate (float): sampling rate in Hz
        algo (str, optional): clustering algorithm. Defaults to 'Leiden'.
        chs (list, optional): channel indices to process. Defaults to [0].
        channel_of_study (int, optional): main channel index for processing. Defaults to 0.
        notchcut, lowcut, highcut: filter parameters
        distance: distance metric
        Other parameters: clustering algorithm parameters

    Returns:
        tuple: (templates_list, frames_list) where templates and spike frames for each channel
    """
    DataFull = Data_ch.copy()
    median_dato = np.median(DataFull, 1)
    NumFrames = DataFull.shape[0]
    final_templates_tot = []
    frames_final_tot = []
    
    for ch in chs:
        tempo = time.time()
        if np.max(np.abs(DataFull[:, ch])) != 8000:
            '''LEARNING TEMPLATES''' 
            '''25 Channels templates''' #a neuron fires to a distance between 50-100 um and the electrodes have a distance from each other of 42 um; do we consider 25 electrodes? (the center and the 24 around)

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
                clusters_L, G, partition = stratification.LeidenAlgo(D.T, threshold_Leiden=0.95, distance='rho', p_minkowski=p_minkowski, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, SamplingRate=SamplingRate)
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
        print(f'SPIKE SORTING ELECTRODE {channel_of_study}: {time.time()-tempo:.2f}s')
        print('---')
    
    gc.collect()
    return final_templates_tot, frames_final_tot


def LinkChannelsSpksort(results):
    """Link spike clusters across neighboring channels to identify single neurons.
    
    Compares spike templates from neighboring channels to identify spikes from the same
    neuron recorded on multiple electrodes.

    Args:
        results (list): spike sorting results from multiple channels

    Returns:
        array: correlation matrix between templates across channels
    """
    n_chs = len(results)
    common_neuron = np.zeros((n_chs, n_chs))
    
    for i in range(n_chs):
        for j in np.arange(i+1, n_chs):
            if len(results[i][0]) > 0 and len(results[j][0]) > 0:
                for s in range(len(results[i][0])):
                    for k in range(len(results[j][0])):
                        try:
                            corr = pearsonr(results[i][0][s], results[j][0][k])[0]
                            if corr >= 0.95:
                                common_neuron[i][j] += 1
                        except:
                            pass
    
    gc.collect()
    return common_neuron


def WrapperSpikesDetection(args_tuple):
    """Wrapper for parallel processing of spike detection.
    
    Args:
        args_tuple (tuple): arguments for SacchiSpikeSorting

    Returns:
        tuple: (templates, frames)
    """
    final_templates, final_frames = SacchiSpikeSorting(*args_tuple)
    gc.collect()
    return final_templates[0] if len(final_templates) > 0 else [], final_frames[0] if len(final_frames) > 0 else []



