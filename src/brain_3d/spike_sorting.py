"""Spike sorting and detection functions for neural data analysis."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pywt
import math
import scipy
import time
from scipy.signal import find_peaks, butter, filtfilt, correlate
from scipy.stats import pearsonr, zscore
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
    """Select non-redundant features using pairwise correlation.

    The function computes a correlation matrix and iteratively removes
    variables that exceed the specified correlation threshold.

    Args:
        df (pd.DataFrame): Input dataframe with feature columns.
        thresh (float, optional): Correlation threshold above which two
            features are treated as redundant. Defaults to 0.9.
        verbose (bool, optional): If ``True``, print progress information.
            Defaults to ``False``.

    Raises:
        ValueError: If the dataframe contains only one variable.

    Returns:
        list: Sorted list of indices corresponding to retained features.
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
    """Detect threshold-crossing peaks in segmented signal windows.

    Detection is performed per window by estimating a robust noise level
    and searching for either positive or negative peaks.

    Args:
        data (array-like): One-dimensional signal.
        step (int): Window length used to compute local baseline/noise.
        threshold (float): Detection threshold multiplier in sigma units.
        aux_spike (str): Peak polarity selector, either ``"pos"`` or ``"neg"``.

    Returns:
        list: Detected spike frame indices in the original signal.
    """
    data = data[:data.shape[0] - (data.shape[0] % step)]
    data_reshaped = data.reshape(-1, step)
    mu = np.mean(data_reshaped, axis=1)
    data_reshaped_aux = data_reshaped - mu[:, np.newaxis]
    sigma = np.median(np.abs(data_reshaped_aux), axis = 1)/0.6745
    th_sigma = threshold * sigma
    
    frames = []
    for ii in range(data_reshaped_aux.shape[0]):
        row = data_reshaped_aux[ii, :]
        if aux_spike == "pos":
            peaks, properties = find_peaks(row, height=th_sigma[ii])
        else:
            peaks, properties = find_peaks(-row, height=th_sigma[ii])
        if len(peaks) > 0:
            #peaks = peaks[np.argmax(properties["peak_heights"])]
            peaks = peaks + step * ii
            for peak in peaks:
                frames.append(peak)    

    return frames            
            
            
def WrapperSpikesDetection(args_tuple):            
    """Run polarity-specific spike detection on one channel.

    The function filters a channel, detects negative and positive events,
    resolves close opposite-polarity candidates, and extracts waveform
    snippets around detected peaks.

    Args:
        args_tuple (tuple): Tuple containing
            ``(DataChannel, ch, parameter, SamplingRate, lowcut, highcut, notchcut)``.

    Returns:
        tuple: ``(frames_N, frames_P, dataset_N, dataset_P)`` where
            ``frames_N`` and ``frames_P`` are negative/positive peak indices,
            and ``dataset_N``/``dataset_P`` are aligned waveform snippets.
    """
    
    DataChannel, ch, parameter, SamplingRate, lowcut, highcut, notchcut = args_tuple
    DataChannel = DataChannel-np.mean(DataChannel)
    DataChannel = brw_f.NotchFilterAlt(DataChannel, notchcut, SamplingRate)
    DataChannel = brw_f.BandpassFilter(DataChannel, lowcut, highcut, SamplingRate)

    if np.abs(np.max(DataChannel)-np.min(DataChannel))<1000:
        DataChannel = DataChannel-np.mean(DataChannel)
        step = int(SamplingRate*0.5) 
        idxnnumber = np.int16(np.ceil(0.002*SamplingRate)+1)
        half_idxnumber = np.int16(np.floor(idxnnumber/2))
        NumFrames = len(DataChannel)

        frames_N_1 = SpikesDetection(DataChannel, step, parameter, "neg")
        frames_P_1 = SpikesDetection(DataChannel, step,parameter, "pos")
        frames_N_2 = np.array(SpikesDetection(DataChannel[int(step/2):-1], step, parameter, "neg"))
        frames_P_2 = np.array(SpikesDetection(DataChannel[int(step/2):-1], step,parameter, "pos"))

        if len(frames_N_2)>0:
            frames_N_2 = frames_N_2 + int(step/2)
        if len(frames_P_2)>0:
            frames_P_2 = frames_P_2 + int(step/2)

        frames_N = np.array(sorted(set(frames_N_1)|set(frames_N_2)))
        frames_P = np.array(sorted(set(frames_P_1)|set(frames_P_2)))

        frames_P_new = []
        frames_N_new = set(frames_N)
        l=0
        while l < len(frames_P):
            if len(set(np.array(range(frames_P[l]-half_idxnumber, frames_P[l]))) & set(frames_N_new))>0:
                f = sorted(set(np.array(range(frames_P[l]-half_idxnumber, frames_P[l]))) & set(frames_N_new))[-1]
                if np.abs(DataChannel[f])<np.abs(DataChannel[frames_P[l]]):
                    if f in set(frames_N_new):
                        frames_N_new.remove(f)
                else:
                    l=l+1
            elif  len(set(np.array(range(frames_P[l], frames_P[l]+half_idxnumber+1))) & set(frames_N_new))>0:
                f = sorted(set(np.array(range(frames_P[l], frames_P[l]+half_idxnumber+1))) & set(frames_N_new))[0]
                if np.abs(DataChannel[f])<np.abs(DataChannel[frames_P[l]]):
                    frames_P_new.append(frames_P[l])
                    if f in set(frames_N_new):
                        frames_N_new.remove(f)
                l=l+1
            else:
                frames_P_new.append(frames_P[l])
                l=l+1
        frames_P = sorted(frames_P_new)
        frames_N = sorted(frames_N_new)

        dataset_N = np.zeros((len(frames_N), idxnnumber)) 
        for k in range(len(frames_N)):
            peak_frame = frames_N[k] 
            if peak_frame < half_idxnumber :
                dataset_N[k, half_idxnumber-peak_frame:idxnnumber] = DataChannel[0:peak_frame+half_idxnumber+1] 
            elif peak_frame >= NumFrames-half_idxnumber :
                aa =  DataChannel[peak_frame-half_idxnumber :NumFrames] 
                dataset_N[k, 0: len(aa) ] = aa
            else:
                dataset_N[k] = DataChannel[peak_frame-half_idxnumber :peak_frame+half_idxnumber +1]

        dataset_P = np.zeros((len(frames_P), idxnnumber)) 
        for k in range(len(frames_P)):
            peak_frame = frames_P[k] 
            if peak_frame < half_idxnumber :
                dataset_P[k, half_idxnumber-peak_frame:idxnnumber] = DataChannel[0:peak_frame+half_idxnumber+1] 
            elif peak_frame >= NumFrames-half_idxnumber :
                aa =  DataChannel[peak_frame-half_idxnumber :NumFrames] 
                dataset_P[k, 0: len(aa) ] = aa
            else:
                dataset_P[k] = DataChannel[peak_frame-half_idxnumber :peak_frame+half_idxnumber +1]
    else: 
        frames_N = [] 
        frames_P = []
        dataset_N = [] 
        dataset_P = [] 
    
    for var in list(locals()):
        if var != 'frames_N' and var != 'frames_P' and var!='dataset_N' and var!='dataset_P':
            del locals()[var]
    gc.collect()


    return frames_N, frames_P, dataset_N, dataset_P    
            
def WrapperTemplateNeg(args_tuple):
    """Learn negative and positive templates for a single electrode.

    This wrapper performs spike detection, channel filtering, and template
    extraction for both polarities, then returns a merged template set.

    Args:
        args_tuple (tuple): Tuple containing
            ``(DataChannel, ch, parameter, algo, distance, method_HC,``
            ``criterion_HC, method_KM, max_iter_FCM, threshold_variance,``
            ``w_max, g, epsilon_EDR, epsilon_LCSS, fuzzy_parameter, noise,``
            ``threshold_dendrogram, max_classes, threshold_Leiden,``
            ``p_minkowski, SamplingRate, normalization, norm_mode,``
            ``notchcut, lowcut, highcut)``.

    Returns:
        tuple: ``(frames_N, frames_P, templates)`` where ``templates``
            contains both negative and positive template waveforms.
    """
    
    tt=time.time()
    DataChannel, ch, parameter, algo, distance, method_HC, criterion_HC, method_KM, max_iter_FCM, threshold_variance, w_max, g, epsilon_EDR, epsilon_LCSS, fuzzy_parameter, noise, threshold_dendrogram, max_classes, threshold_Leiden, p_minkowski, SamplingRate, normalization, norm_mode, notchcut, lowcut, highcut = args_tuple
    print('Start '+str(ch))
    epsilon_EDR = np.std(DataChannel)/2
    epsilon_LCSS = np.std(DataChannel)/2
    frames_N, frames_P, dataset_N, dataset_P = WrapperSpikesDetection((DataChannel, ch, parameter, SamplingRate, lowcut, highcut, notchcut))

    DataChannel = DataChannel-np.mean(DataChannel)
    DataChannel = brw_f.NotchFilterAlt(DataChannel, notchcut, SamplingRate)
    DataChannel = brw_f.BandpassFilter(DataChannel, lowcut, highcut, SamplingRate)

    templates_N = TemplateNeg(DataChannel, frames_N, SamplingRate, parameter, algo, distance, method_HC, criterion_HC, method_KM, max_iter_FCM, threshold_variance, w_max, g, epsilon_EDR, epsilon_LCSS, fuzzy_parameter, noise, threshold_dendrogram, max_classes, threshold_Leiden, p_minkowski, normalization=normalization, norm_mode=norm_mode) [1]
    templates_P = TemplateNeg(-DataChannel, frames_P, SamplingRate, parameter, algo, distance, method_HC, criterion_HC, method_KM, max_iter_FCM, threshold_variance, w_max, g, epsilon_EDR, epsilon_LCSS, fuzzy_parameter, noise, threshold_dendrogram, max_classes, threshold_Leiden, p_minkowski, normalization=normalization, norm_mode=norm_mode) [1]
    templates_P = -templates_P
    templates = []
    
    for n in range(len(templates_N.T)):
        templates.append(templates_N[:,n])
    
    for p in range(len(templates_P.T)):
        templates.append(templates_P[:,p])
    templates=np.array(templates)

    print('Spikes detection & Templates learning electrode '+str(ch)+': '+str(time.time()-tt))

    for var in list(locals()):
        if var != 'frames_N' and var != 'templates' and var != 'frames_P':
            del locals()[var]
    gc.collect()

    return frames_N, frames_P, templates
           
def TemplateNeg(DataChannel, frames_N, SamplingRate, parameter = 4.5, algo = 'Leiden', distance = 'rho', method_HC = 'complete', criterion_HC = 'distance', method_KM = 'silhouette', max_iter_FCM=10, threshold_variance = 0.9, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, fuzzy_parameter = 1, noise = 0, threshold_dendrogram = 0.33, max_classes = [2], threshold_Leiden = 0.9, p_minkowski = 2, frequency=1000, normalization = 'OFF', norm_mode ='min_max_single'):
    """Learn candidate spike templates for one channel via clustering.

    Builds fixed-length waveform snippets from pre-detected spike frames,
    clusters them, merges highly correlated templates, and retains only
    templates that satisfy morphology constraints.

    Args:
        DataChannel (array-like): One-dimensional filtered signal for one channel.
        frames_N (list): Pre-detected spike frame indices.
        SamplingRate (float): Sampling frequency in Hz.
        parameter (float): spikes detection parameter. Default to 4.5
        algo (string): clustering algorithm'. Default to 'Leiden'.
        distance (str, optional): metric for clustering. Defaults to 'rho'.
        method_HC (str, optional): linkage method. Defaults to 'complete'.
        criterion_HC (str, optional): hierarchical clustering criterion. Defaults to 'distance'.
        method_KM (str, optional): method to compute the optimal number of centroids in KM and FCM or relatives. Defaults to 'silhouette'.
        max_iter_FCM (int, optional): maximum number of iterations in FCM. Defaults to 10.
        threshold_variance (float, optional): explained variance after PCA. Defaults to 0.9.
        w_max (int, optional): WDTW and WDDTW parameter. Defaults to 1.
        g (int, optional): WDTW and WDDTW parameter. Defaults to 1.
        epsilon_EDR (float, optional): EDR threshold. Defaults to 0.001.
        epsilon_LCSS (float, optional): LCSS threshold. Defaults to 0.001.
        fuzzy_parameter (int, optional): FCM parameter. Defaults to 1.
        noise (int, optional): amount of noise in percentage to add to data. Defaults to 0.
        threshold_dendrogram (float, optional): cut height of the dendrogram. Defaults to 0.33.
        max_classes (int, optional): maximum number of classes for clustering. Defaults to 1.
        threshold_Leiden (float, optional): Leiden threshold. Defaults to 0.9.
        p_minkowski (int, optional): Minkowski parameter. Defaults to 2.
        frequency (float, optional): Sampling frequency passed to the clustering
            module (alias of ``SamplingRate``). Defaults to ``1000``.
        normalization (str, optional): Normalization mode switch. Defaults to ``'OFF'``.
        norm_mode (str, optional): Normalization strategy. Defaults to ``'min_max_single'``.

    Returns:
        tuple: ``(clusters, templates_N, frames_N)`` where ``clusters`` is
            ``[n_clusters, cluster_indices]``, ``templates_N`` is the template
            matrix ``[n_samples, n_templates]``, and ``frames_N`` is the
            list of input spike frames.
    """
 
    idxnnumber = np.int16(np.ceil(0.002*SamplingRate)+1)
    half_idxnumber = np.int16(np.floor(idxnnumber/2))
    NumFrames = len(DataChannel)
    dataset_N = np.zeros((len(frames_N), idxnnumber)) 
    for k in range(len(frames_N)):
        peak_frame = frames_N[k] 
        if peak_frame < half_idxnumber :
            dataset_N[k, half_idxnumber-peak_frame:idxnnumber] = DataChannel[0:peak_frame+half_idxnumber+1] 
        elif peak_frame >= NumFrames-half_idxnumber :
            aa =  DataChannel[peak_frame-half_idxnumber :NumFrames] 
            dataset_N[k, 0: len(aa) ] = aa
        else:
            dataset_N[k] = DataChannel[peak_frame-half_idxnumber :peak_frame+half_idxnumber +1]
    dataset_N_aux = dataset_N.copy()
    if dataset_N_aux.shape[0]>1:
        clusters = stratification.RecursiveClustering(data=dataset_N_aux, algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, SamplingRate=frequency, p_minkowski=p_minkowski, normalization=normalization, norm_mode=norm_mode)
        # clusters = clustering.Clustering(data=dataset_N_aux, algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, SamplingRate=frequency, p_minkowski=p_minkowski, normalization=normalization, norm_mode=norm_mode)

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
        for c in range(clusters[0]):
            peaks_N = scipy.signal.find_peaks(-templates_N[:,c]) [0]# peaks_N = scipy.signal.find_peaks(-templates_N[:,c], prominence=np.abs(templates_N[20,c])/10) [0]
            # '''
            prominences_N = scipy.signal.peak_prominences(-templates_N[:,c], peaks_N)[0]
            largeness_N = scipy.signal.peak_widths(-templates_N[:,c], peaks_N, rel_height = 1)[0]
            peaks_N_new = list(peaks_N)
            for p in range(len(peaks_N)):
                if largeness_N[p]>=prominences_N[p]:
                    peaks_N_new.remove(peaks_N[p])
            peaks_N = np.array(peaks_N_new)
            # '''
            peaks_P = scipy.signal.find_peaks(templates_N[:,c]) [0] #peaks_P = scipy.signal.find_peaks(templates_N[:,c], prominence=np.abs(templates_N[20,c])/10) [0]
            prominences_P = scipy.signal.peak_prominences(templates_N[:,c], peaks_P)[0]
            largeness_P = scipy.signal.peak_widths(templates_N[:,c], peaks_P, rel_height = 1)[0]
            peaks_P_new = list(peaks_P)
            for p in range(len(peaks_P)):
                if largeness_P[p]>=prominences_P[p]:
                    peaks_P_new.remove(peaks_P[p])
            peaks_P_new = np.array(peaks_P_new)
            if len(peaks_N)==1 and len(clusters[1][c])>4 and len(peaks_P_new[(peaks_P_new<half_idxnumber) & (peaks_P_new>half_idxnumber/2)])<=1 and len(peaks_P_new[(peaks_P_new<half_idxnumber/2)])==0 and len(peaks_P_new[(peaks_P_new>half_idxnumber) & (peaks_P_new<half_idxnumber/2+half_idxnumber)])<=1 and len(peaks_P_new[(peaks_P_new>half_idxnumber/2+half_idxnumber)])==0: 
                clusters_N.append(clusters[1][c])
                templates.append(templates_N[:, c])
                frames_new = set(frames_new)|set(np.array(frames_N)[clusters[1][c]])
            elif len(peaks_N)>1 and len(peaks_N[peaks_N<half_idxnumber])<=1 and len(peaks_N[peaks_N>half_idxnumber])<=1 and len(clusters[1][c])>4 and len(peaks_P_new[(peaks_P_new<half_idxnumber) & (peaks_P_new>half_idxnumber/2)])<=1 and len(peaks_P_new[(peaks_P_new<half_idxnumber/2)])==0 and len(peaks_P_new[(peaks_P_new>half_idxnumber) & (peaks_P_new<half_idxnumber/2+half_idxnumber)])<=1 and len(peaks_P_new[(peaks_P_new>half_idxnumber/2+half_idxnumber)])==0: 
                idxs = list(np.arange(len(peaks_N)))
               # if len(np.where(peaks_N==20)[0])>0: #vedere perché ogni tanto non viene preso il 20 
                idx_20 = np.where(peaks_N==half_idxnumber)[0][0]
                idxs.remove(idx_20)
                cont = 0
                peaks_P = np.array(peaks_P_new)
                for idx in idxs:
                    if scipy.signal.peak_widths(-templates_N[:,c], peaks_N, rel_height = 1)[0][idx]<=np.abs(scipy.signal.peak_prominences(-templates_N[:,c], peaks_N)[0][idx]):
                        cont+=1
                if cont == 0 and len(clusters[1][c])>2:
                    clusters_N.append(clusters[1][c])
                    templates.append(templates_N[:, c])
                    frames_new = set(frames_new)|set(np.array(frames_N)[clusters[1][c]])
        
    
    elif dataset_N_aux.shape[0]==0:
        clusters_N = [] 
        templates = []
        frames_new = [] 
    else: 
        clusters = (1, [[0]])
        templates_N = dataset_N_aux.copy().T
        clusters_N = [] 
        templates = []  
        frames_new = ()
        for c in range(clusters[0]):
            peaks_N = scipy.signal.find_peaks(-templates_N[:,c]) [0]# peaks_N = scipy.signal.find_peaks(-templates_N[:,c], prominence=np.abs(templates_N[20,c])/10) [0]
            # '''
            prominences_N = scipy.signal.peak_prominences(-templates_N[:,c], peaks_N)[0]
            largeness_N = scipy.signal.peak_widths(-templates_N[:,c], peaks_N, rel_height = 1)[0]
            peaks_N_new = list(peaks_N)
            for p in range(len(peaks_N)):
                if largeness_N[p]>=prominences_N[p]:
                    peaks_N_new.remove(peaks_N[p])
            peaks_N = np.array(peaks_N_new)
            # '''
            peaks_P = scipy.signal.find_peaks(templates_N[:,c]) [0] #peaks_P = scipy.signal.find_peaks(templates_N[:,c], prominence=np.abs(templates_N[20,c])/10) [0]
            prominences_P = scipy.signal.peak_prominences(templates_N[:,c], peaks_P)[0]
            largeness_P = scipy.signal.peak_widths(templates_N[:,c], peaks_P, rel_height = 1)[0]
            peaks_P_new = list(peaks_P)
            for p in range(len(peaks_P)):
                if largeness_P[p]>=prominences_P[p]:
                    peaks_P_new.remove(peaks_P[p])
            peaks_P_new = np.array(peaks_P_new)
            if len(peaks_N)==1 and len(clusters[1][c])>4 and len(peaks_P_new[(peaks_P_new<20) & (peaks_P_new>half_idxnumber/2)])<=1 and len(peaks_P_new[(peaks_P_new<half_idxnumber/2)])==0 and len(peaks_P_new[(peaks_P_new>half_idxnumber) & (peaks_P_new<half_idxnumber/2+half_idxnumber)])<=1 and len(peaks_P_new[(peaks_P_new>half_idxnumber/2+half_idxnumber)])==0: 
                clusters_N.append(clusters[1][c])
                templates.append(templates_N[:, c])
                frames_new = set(frames_new)|set(np.array(frames_N)[clusters[1][c]])
            elif len(peaks_N)>1 and len(peaks_N[peaks_N<20])<=1 and len(peaks_N[peaks_N>20])<=1 and len(clusters[1][c])>4 and len(peaks_P_new[(peaks_P_new<20) & (peaks_P_new>half_idxnumber/2)])<=1 and len(peaks_P_new[(peaks_P_new<half_idxnumber/2)])==0 and len(peaks_P_new[(peaks_P_new>20) & (peaks_P_new<half_idxnumber/2+half_idxnumber)])<=1 and len(peaks_P_new[(peaks_P_new>half_idxnumber/2+half_idxnumber)])==0: 
                idxs = list(np.arange(len(peaks_N)))
               # if len(np.where(peaks_N==20)[0])>0: #vedere perché ogni tanto non viene preso il 20 
                idx_20 = np.where(peaks_N==20)[0][0]
                idxs.remove(idx_20)
                cont = 0
                peaks_P = np.array(peaks_P_new)
                for idx in idxs:
                    if scipy.signal.peak_widths(-templates_N[:,c], peaks_N, rel_height = 1)[0][idx]<=np.abs(scipy.signal.peak_prominences(-templates_N[:,c], peaks_N)[0][idx]):
                        cont+=1
                if cont == 0 and len(clusters[1][c])>2:
                    clusters_N.append(clusters[1][c])
                    templates.append(templates_N[:, c])
                    frames_new = set(frames_new)|set(np.array(frames_N)[clusters[1][c]])
                    
    templates_N = np.array(templates).T

    clusters = [len(clusters_N), clusters_N] 

    return clusters, templates_N, list(frames_N)

def TemplateMatching(data, templates, thresh = 0.95):
    """Match learned templates against a signal and subtract detections.

    For each template, normalized cross-correlation is computed along the
    signal. Candidate matches above threshold are assigned to the best
    scoring template and removed from the signal.

    Args:
        data (array-like): One-dimensional signal to analyze.
        templates (array-like): Template matrix with shape
            ``(n_templates, n_samples_per_template)``.
        thresh (float, optional): Minimum normalized correlation required
            to accept a match. Defaults to ``0.95``.

    Returns:
        tuple: ``(frames, data, dataset, dataset_idx)`` where ``frames`` are
            match indices per template, ``data`` is the residual signal,
            ``dataset`` contains extracted matched waveforms, and
            ``dataset_idx`` maps each template to items in ``dataset``.
    """

    n = templates.shape[0]
    if n==0:
        return [], data, [], []
    else:
        size = int(templates.shape[1]/2)
        data_aux = data.copy() 
        data_aux = np.asarray(data_aux)
        data_cumsum = np.cumsum(np.insert(data_aux,0,0))
        data_cumsum2 = np.cumsum(np.insert(data_aux**2,0,0))
        window_sum = data_cumsum[templates.shape[1]:] - data_cumsum[:-templates.shape[1]]
        window_sum2 = data_cumsum2[templates.shape[1]:] - data_cumsum2[:-templates.shape[1]]  
        window_mean = window_sum/templates.shape[1]
        window_std = np.sqrt((window_sum2-(window_sum**2)/templates.shape[1])/templates.shape[1])

        half_idxnumber = np.int16(np.floor(templates.shape[1]/2))
        
        frames = []
        dataset_idx = [] 
        for c in range(n):
            frames.append([])
            dataset_idx.append([])
        dataset = []
        '''Spikes detection by templates matching'''
        templates_aux = templates.copy()
        correlations =[] 
        indexes =[] 
        indexes_glob = set() 
        for i in range(n):
            template = np.asarray(templates_aux[i]) 
            template_norm = (template-np.mean(template))/np.std(template)
            raw_corr = correlate(data_aux, template_norm, mode='valid')
            norm_corr = raw_corr/(window_std*templates.shape[1])
            correlations.append(norm_corr)
            positions = scipy.signal.find_peaks(norm_corr, height = thresh)[0]+size
            indexes.append(positions)
            indexes_glob = indexes_glob | set(positions) 
        for idx in sorted(indexes_glob):
            corr_values=0
            for i in range(n):
                if idx in indexes[i] and correlations[i][idx-size]>corr_values:
                    corr_values = correlations[i][idx-size]
                    id_template_to_subtract = i
            frames[id_template_to_subtract].append(idx)
            dataset.append(templates[id_template_to_subtract]/np.linalg.norm(templates[id_template_to_subtract])*np.linalg.norm(data[idx-half_idxnumber :idx+half_idxnumber +1])) 
            dataset_idx[id_template_to_subtract].append(len(dataset)-1)    
            data[idx-size:idx+size+1] = data[idx-size:idx+size+1]-templates[id_template_to_subtract]/np.linalg.norm(templates[id_template_to_subtract])*np.linalg.norm(data[idx-half_idxnumber :idx+half_idxnumber +1])
        return frames, data, dataset, dataset_idx

def WrapperSpikeSorting(args_tuple):
    """Execute spike sorting and collect execution metadata.

    Args:
        args_tuple (tuple): Positional arguments forwarded to ``SpikeSorting``.

    Returns:
        tuple: ``(final_templates, final_frames, time_to_save,
        MAX_BEFORE_MATCHING, MAX_AFTER_MATCHING,
        MIN_BEFORE_MATCHING, MIN_AFTER_MATCHING)``.
    """

    t0 = time.time()
    final_templates, final_frames = SpikeSorting(*args_tuple)
    time_to_save = time.time() - t0

    MAX_BEFORE_MATCHING = None
    MIN_BEFORE_MATCHING = None
    if len(args_tuple) > 0:
        data_in = np.asarray(args_tuple[0])
        if data_in.size > 0:
            MAX_BEFORE_MATCHING = float(np.max(data_in))
            MIN_BEFORE_MATCHING = float(np.min(data_in))

    MAX_AFTER_MATCHING = None
    MIN_AFTER_MATCHING = None
    gc.collect()

    return final_templates[0], final_frames[0], time_to_save, MAX_BEFORE_MATCHING, MAX_AFTER_MATCHING, MIN_BEFORE_MATCHING, MIN_AFTER_MATCHING


def ChannelsSpksort(ch, nrow=64, ncol=64):
    """Return the local 5x5 channel neighborhood in a grid layout.

    Args:
        ch (int): Linear index of the central channel.
        nrow (int, optional): Number of grid rows. Defaults to ``64``.
        ncol (int, optional): Number of grid columns. Defaults to ``64``.

    Returns:
        tuple: ``(chs, idx_ch)`` where ``chs`` is the sorted neighborhood
            channel array and ``idx_ch`` is the position of ``ch`` in ``chs``.
    """
    row = ch//nrow
    col = ch % ncol
    rows = np.arange(row-2,row+2+1)
    cols = np.arange(col-2, col+2+1)
    rows = rows[rows>=0]
    rows = rows[rows<=nrow-1]
    cols = cols[cols>=0]
    cols = cols[cols<=ncol-1]
    chs = []
    for i in rows:
        for j in cols:
            chs.append(i*ncol+j)
    chs = np.array(sorted(chs))
    chs = chs[chs>=0]
    chs = chs[chs<=nrow*ncol-1]  
    idx_ch = np.where(chs==ch)[0]

    gc.collect()

    return chs, idx_ch

def SpikeSorting(Data_ch, SamplingRate, algo='Leiden', chs=[0], channel_of_study=0, 
                      notchcut=50, lowcut=300, highcut=3000, distance='rho', 
                      method_HC='complete', criterion_HC='distance', method_KM='silhouette', 
                      max_iter_FCM=10, threshold_variance=0.9, w_max=1, g=1, 
                      epsilon_EDR=0.001, epsilon_LCSS=0.001, fuzzy_parameter=1, noise=0, 
                      threshold_dendrogram=0.33, max_classes=[2], threshold_Leiden=0.9, 
                      p_minkowski=2, normalization='OFF', norm_mode='min_max_single'):
    """Run the full unsupervised spike-sorting workflow.

    The pipeline performs channel filtering, template learning, iterative
    template matching/subtraction, graph-based clustering, and reconstruction
    of assigned events.

    Args:
        Data_ch (array-like): Signal matrix with shape ``[samples, channels]``.
        SamplingRate (float): Sampling frequency in Hz.
        algo (str, optional): Clustering backend. Defaults to ``'Leiden'``.
        chs (list, optional): Channel indices to process. Defaults to ``[0]``.
        channel_of_study (int, optional): Channel label used in logs.
            Defaults to ``0``.
        notchcut (float, optional): Notch filter frequency.
        lowcut (float, optional): Band-pass lower cutoff.
        highcut (float, optional): Band-pass upper cutoff.
        distance (str, optional): Distance metric used in clustering.
        method_HC (str, optional): Hierarchical clustering linkage method.
        criterion_HC (str, optional): Hierarchical clustering cut criterion.
        method_KM (str, optional): Cluster-selection strategy.
        max_iter_FCM (int, optional): Maximum FCM iterations.
        threshold_variance (float, optional): Variance threshold.
        w_max (float, optional): Maximum warping window/weight parameter.
        g (float, optional): Distance-model parameter.
        epsilon_EDR (float, optional): EDR tolerance parameter.
        epsilon_LCSS (float, optional): LCSS tolerance parameter.
        fuzzy_parameter (float, optional): FCM fuzzifier.
        noise (float, optional): Noise handling parameter.
        threshold_dendrogram (float, optional): Dendrogram cut threshold.
        max_classes (list, optional): Maximum classes per recursive split.
        threshold_Leiden (float, optional): Leiden similarity threshold.
        p_minkowski (float, optional): Minkowski distance order.
        normalization (str, optional): Normalization mode switch.
        norm_mode (str, optional): Normalization strategy.

    Returns:
        tuple: ``(final_templates_tot, frames_final_tot)`` for processed channels.
    """
    DataFull = Data_ch.copy()
    median_dato = np.median(DataFull, 1)
    NumFrames = DataFull.shape[0]
    final_templates_tot = []
    frames_final_tot = []
    
    idxnnumber = np.int16(np.ceil(0.002*SamplingRate)+1)
    half_idxnumber = np.int16(np.floor(idxnnumber/2))
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
                DataFull[:, channel] = brw_f.NotchFilterAlt(DataFull[:, channel], notchcut, SamplingRate)
                DataFull[:, channel] = brw_f.BandpassFilter(DataFull[:, channel], lowcut, highcut, SamplingRate)
                DataFull[:, channel] = DataFull[:, channel] - np.mean(DataFull[:, channel])
                if np.max(np.abs(DataFull[:,channel]))!=8000:
                    step_det = int(SamplingRate*0.5)
                    frames_P_ch = SpikesDetection(DataFull[:, channel], step_det, 4.5, "pos")
                    frames_N_ch = SpikesDetection(DataFull[:, channel], step_det, 4.5, "neg")
                    clusters_P, templates_P, frames_P = TemplateNeg(-DataFull[:, channel], frames_P_ch, SamplingRate, parameter=4.5, algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, p_minkowski=p_minkowski, frequency=SamplingRate)
                    clusters_N, templates_N, frames_N = TemplateNeg(DataFull[:, channel], frames_N_ch, SamplingRate, parameter=4.5, algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, p_minkowski=p_minkowski, frequency=SamplingRate)
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

            if templates.shape[0]>1:
                df = pd.DataFrame(templates.T)
                columns = FindCorrelation(df, thresh = 0.95)
                templates = templates[columns,:]  
  

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
                step_det = int(SamplingRate*0.5)
                frames_P_iter = SpikesDetection(DataFull[:, ch], step_det, par[p], "pos")
                frames_N_iter = SpikesDetection(DataFull[:, ch], step_det, par[p], "neg")
                clusters_P, templates_P, frames_P = TemplateNeg(-DataFull[:, ch], frames_P_iter, SamplingRate, parameter=par[p], algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, p_minkowski=p_minkowski, frequency=SamplingRate)
                clusters_N, templates_N, frames_N = TemplateNeg(DataFull[:, ch], frames_N_iter, SamplingRate, parameter=par[p], algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, p_minkowski=p_minkowski, frequency=SamplingRate)
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
                        if i>=half_idxnumber and i<NumFrames-half_idxnumber:
                            if i in np.array(sorted(set(frames_N))) and len(templates_neg)>0:
                                corr = []
                                for c in range(len(signals)):
                                    corr.append(pearsonr(signals[c]/np.linalg.norm(signals[c]), DataFull[i-half_idxnumber:i+half_idxnumber+1, ch]/np.linalg.norm(DataFull[i-half_idxnumber:i+half_idxnumber+1, ch]))[0])
                                corr = np.array(corr)
                                if np.max(corr)>0.5:
                                    idx = np.where(corr==np.max(corr))[0][0]
                                    frames_tot[idx].append(i)
                                    dataset_idx_tot[idx].append(len(dataset)) 
                                    dataset.append(signals[idx]/np.linalg.norm(signals[idx])*np.linalg.norm(DataFull[i-half_idxnumber:i+half_idxnumber+1, ch]))
                                    DataFull[i-half_idxnumber:i+half_idxnumber+1, ch] = DataFull[i-half_idxnumber:i+half_idxnumber+1, ch]-signals[idx]/np.linalg.norm(signals[idx])*np.linalg.norm(DataFull[i-half_idxnumber:i+half_idxnumber+1, ch])
                            if i in np.array(sorted(set(frames_P))) and len(templates_pos)>0:
                                corr = [] 
                                for c in range(len(signals)):
                                    corr.append(pearsonr(signals[c]/np.linalg.norm(signals[c]), DataFull[i-half_idxnumber:i+half_idxnumber+1, ch]/np.linalg.norm(DataFull[i-half_idxnumber:i+half_idxnumber+1, ch]))[0])
                                corr = np.array(corr)
                                if np.max(corr)>0.5:
                                    idx = np.where(corr==np.max(corr))[0][0]
                                    frames_tot[idx].append(i) 
                                    dataset_idx_tot[idx].append(len(dataset))
                                    dataset.append(signals[idx]/np.linalg.norm(signals[idx])*np.linalg.norm(DataFull[i-half_idxnumber:i+half_idxnumber+1, ch]))
                                    DataFull[i-half_idxnumber:i+half_idxnumber+1, ch] = DataFull[i-half_idxnumber:i+half_idxnumber+1, ch]-signals[idx]/np.linalg.norm(signals[idx])*np.linalg.norm(DataFull[i-half_idxnumber:i+half_idxnumber+1, ch])
                            
                    p=p+1

            # print('Templates matching: '+str(time.time()-t))
            
            n_tot = len(dataset)
            D = np.zeros((n_tot, idxnnumber)) 
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

            final_templates = np.zeros((len(clusters), idxnnumber))
            #legend_graph=[]
            frames_final = []
            #plt.figure()
            #plt.title('Neurons shapes candidate')
            for c in range(len(clusters)):
                final_templates[c] = np.mean(D[clusters[c]].T,1).reshape(idxnnumber)
                frames_final.append([])
      
            '''Merging Tree'''
            if len(clusters)>1:
                # t = time.time()
                merge.MergingTree(G, partition)
                # print('Merging Tree: '+str(time.time()-t))

            '''Reconstruction''' 
            reconstruction = np.zeros(NumFrames)
            
            for c in range(len(frames_tot)):
                for i in range(len(frames_tot[c])):
                    f = frames_tot[c][i]
                    aux_idx = dataset_idx_tot[c][i]
                    high_peak = dataset[aux_idx][half_idxnumber]
                    j=0
                    while j < (len(final_templates)):
                        if aux_idx in clusters[j]:
                            reconstruction[f-half_idxnumber:f+half_idxnumber+1]+=final_templates[j]/final_templates[j][half_idxnumber]*high_peak
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






































