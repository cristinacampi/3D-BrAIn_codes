"""
Codes with functions related to Spikes sorting
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from brain_3d import stratification
import brw_functions as brw_f
import time
from scipy.signal import find_peaks, butter, filtfilt
from statistics import median
from scipy.stats import pearsonr
import igraph as ig
from igraph import Graph, plot
import leidenalg as la
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
    Select a subset of variables by removing highly correlated features.

    The function computes the correlation matrix of the input DataFrame and
    iteratively removes variables whose correlation exceeds the specified
    threshold, returning the indices of the retained columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing the variables to analyze.
    thresh : float, optional
        Correlation threshold above which two variables are considered
        redundant. Default is 0.9.
    verbose : bool, optional
        If True, prints diagnostic information during the selection process.
        Default is False.

    Returns
    -------
    list
        Sorted list of column indices retained after correlation filtering.

    Raises
    ------
    ValueError
        If the input DataFrame contains only one variable.
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

def SpikesDetection(Data , step, threshold, aux_spike): 
    """
    Spikes detection on negative peaks: we have a spike when the peak is lower than a threshold t = -mu-threshold*sigma (mu is the mean of the signal and sigma is the standard deviation)
    
    Returns:
        frames (np.ndarray): list of frames indexes where a negative spike is detected.
    """
    Data =  Data[:Data .shape[0] - (Data .shape[0] % step)]
    DataReshaped = Data .reshape(-1, step)
    mu = np.mean(DataReshaped,axis=1)
    sigma = np.std(DataReshaped,axis=1)
    DataReshapedAux = DataReshaped-mu[:, np.newaxis] 
    th_sigma = threshold*sigma
    
    frames = []  # per salvare gli indici dei picchi per ogni riga
    for ii in range(DataReshapedAux.shape[0]):
        row = DataReshapedAux[ii, :]
        if aux_spike == "pos":
            peaks, properties = find_peaks(row, height=th_sigma[ii])
        else:
            peaks, properties = find_peaks(-row, height=th_sigma[ii])
        if len(peaks) > 0:
            peaks = peaks[np.argmax(properties["peak_heights"])]
            peaks  = peaks + step*ii
            frames.append(peaks)


    return frames

def TemplateNeg(Data , ch, parameter = 4.5, algo = 'Leiden', distance = 'rho', method_HC = 'complete', criterion_HC = 'distance', method_KM = 'silhouette', max_iter_FCM=10, threshold_variance = 0.9, wMax  = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, FuzzyParameter = 1, noise = 0, ThresholdDendrogram = 0.33, MaxClasses = [2], threshold_Leiden = 0.9, pMinkowski = 2, frequency=1000, Normalization = 'OFF', NormMode ='min_max_single'):
    """
    Negative templates learning
    
    Args:
        Data (np.ndarray): 2D matrix representing the Data set (number of frames x number of channels).
        ch (int): Channel idx (between 0 and number of channels -1) on which we want to learn templates.
        parameter (float): spikes detection parameter. Defaults to 4.5.
        algo (str): clustering algorithm'. Defaults to 'Leiden'.
        distance (str): metric for clustering. Defaults to 'rho'.
        method_HC (str): linkage method. Defaults to 'complete'.
        criterion_HC (str): hierarchical clustering criterion. Defaults to 'distance'.
        method_KM (str): method to compute the optimal number of centroids in KM and FCM or relatives. Defaults to 'silhouette'.
        max_iter_FCM (int): maximum number of iterations in FCM. Defaults to 10.
        threshold_variance (float): explained variance after PCA. Defaults to 0.9.
        wMax  (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilonEDR (float): EDR threshold. Defaults to 0.001.
        epsilonLCSS (float): LCSS threshold. Defaults to 0.001.
        FuzzyParameter (int): FCM parameter. Defaults to 1.
        noise (int): amount of noise in percentage to add to Data . Defaults to 0.
        ThresholdDendrogram (float): cut height of the dendrogram. Defaults to 0.33.
        MaxClasses (int): maximum number of classes for clustering. Defaults to 1.
        threshold_Leiden (float): Leiden threshold. Defaults to 0.9.
        pMinkowski (int): Minkowski parameter. Defaults to 2.
        frequency (int): STS parameter. Defaults to 1000.
        Normalization (str): To applying Normalization. Defaults to 'OFF'.
        NormMode (str): If Normalization applied, to select the modality. Defaults to 'min_max_single'.
    
    Returns:
        clusters (list): number of cluster and clusters obtained from template learning.
        templates_N (np.ndarray): centroids of the clusters.
        frames_N (list): frames idx of all the spikes detected.
    """

    DataChannel =  Data[:, ch].copy()
    NumFrames = DataChannel.shape[0] 
    step = int(frequency*0.01) 
    t=0
    n_frames_ch_neg = 0
    frames_N = {}  
    while(t<NumFrames-step):
        mu = np.mean(Data[t:t+step, ch])
        sigma = np.std(Data[t:t+step, ch]) 
        frames_neg = SpikesDetectionNeg(Data[t:t+step,:] , ch, parameter)+t
        t=t+step
        n_frames_ch_neg += len(frames_neg)
        frames_N = set(frames_N)|set(frames_neg)
    mu = np.mean(Data[t:NumFrames, ch])
    sigma = np.std(Data[t:NumFrames, ch]) 
    frames_neg = SpikesDetectionNeg(Data[t:NumFrames,:] , ch, parameter)+t
    n_frames_ch_neg += len(frames_neg)
    frames_N = set(frames_N)|set(frames_neg)
    frames_N = sorted(frames_N)

    #'''
    t=0
    n_frames_ch_pos = 0
    frames_P = {}  
    while(t<=NumFrames-step):
        mu = np.mean(Data[t:t+step, ch])
        sigma = np.std(Data[t:t+step, ch]) 
        frames_pos = SpikesDetectionNeg(- Data[t:t+step,:] , ch, 4.5)+t
        t=t+step
        n_frames_ch_pos += len(frames_pos)
        frames_P = set(frames_P)|set(frames_pos)
    mu = np.mean(Data[t:NumFrames, ch])
    sigma = np.std(Data[t:NumFrames, ch]) 
    frames_pos = SpikesDetectionNeg(- Data[t:NumFrames,:] , ch, 4.5)+t
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

    Dataset_N = np.zeros((len(frames_N), 41)) 
    for k in range(len(frames_N)):
        peak_frame = frames_N[k] 
        if peak_frame < 20:
            Dataset_N[k, 20-peak_frame:41] = DataChannel[0:peak_frame+21] 
        elif peak_frame >= NumFrames-20:
            Dataset_N[k, 0: NumFrames-peak_frame+20] = DataChannel[peak_frame-20:NumFrames] 
        else:
            Dataset_N[k] = DataChannel[peak_frame-20:peak_frame+21] 
    Dataset_N_aux = Dataset_N.copy()
    if Dataset_N_aux.shape[0]>1:
        clusters = stratification.Recursive_clustering(Data =Dataset_N_aux, algo=algo, distance=distance, method_HC=method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, wMax =wMax , g=g, epsilonEDR=epsilonEDR, epsilonLCSS=epsilonLCSS, FuzzyParameter=FuzzyParameter, noise=noise, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=MaxClasses, threshold_Leiden=threshold_Leiden, SamplingRate=frequency, pMinkowski=pMinkowski, Normalization=Normalization, NormMode=NormMode)

        templates_N =[] 
        for c in range(clusters[0]):
            Data = Dataset_N[clusters[1][c]]
            Data = Data .reshape((Data .shape[-2], Data .shape[-1]))
            mu = np.mean(Data ,0)
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
                Data = Dataset_N[clusters[1][c]]
                Data = Data .reshape((Data .shape[-2], Data .shape[-1]))
                mu = np.mean(Data ,0)
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
            Data = Dataset_N[clusters[1][c]]
            mu = np.mean(Data ,0)
            sigma = np.std(Data ,0) 
            plt.plot(np.arange(Data .shape[1]), mu)
            plt.fill_between(np.arange(Data .shape[1]), mu-sigma, mu+sigma, alpha = 0.2)
            # for k in range(len(clusters[1][c])):
                # plt.plot(np.arange(41), Dataset_N[clusters[1][c][k]]) 
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
        
    
    elif Dataset_N_aux.shape[0]==0:
        clusters_N = [] 
        templates = []
        frames_new = [] 
    else: 
        clusters = (1, [[0]])
        templates_N = Dataset_N_aux.copy()
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

def TemplateMatching(Data , templates, thresh = 0.95):
    """
    Templates matching
    
    Args:
        Data  (np.ndarray): vector representing the signal where compute the templates matching.
        templates (np.ndarray): matrix number of templates x number of frames representing found templates.
        thresh (float): match when Pearson's correlation coefficient upper than the threshold. Defaults to 0.95.
    
    Returns:
        frames (list): frames idx associated to the matching template.
        Data (np.ndarray): signal after matching subtraction.
        Dataset (list): matching frames waveforms.
        DatasetIdx (list): association between Data set element and matching templates.
    """

    n = templates.shape[0]
    if n==0:
        return [], Data , [], []
    else:
        size = int(templates.shape[1]/2)
        x = set(np.arange(Data .shape[0]))
        x_1 = set(np.arange(size))  
        x_2 = set(np.arange(size+4)+Data .shape[0]-size-4) #5 generico 
        y = np.array(sorted(x-x_1-x_2))
        frames = []
        DatasetIdx = [] 
        for c in range(n):
            frames.append([])
            DatasetIdx.append([])
        Dataset = []
        '''Spikes detection by templates matching'''
        i = y[0]   
        while i <= y[-1] :
            #t=time.time()
            corr = [] 
            for c in range(n):
                corr.append(pearsonr(templates[c]/np.linalg.norm(templates[c]),  Data[i-20:i+21]/np.linalg.norm(Data[i-20:i+21]))[0])
            corr = np.array(corr)
            if np.max(corr)>thresh:
                idx = np.where(corr==np.max(corr))[0][0]
                corr_aux = np.zeros(5) 
                for j in range(5):
                    corr_aux[j]= pearsonr(templates[idx]/np.linalg.norm(templates[idx]),  Data[i+j-20:i+j+21]/np.linalg.norm(Data[i+j-20:i+j+21]))[0]
                j_max = np.where(corr_aux==np.max(corr_aux))[0][0]
                frames[idx].append(i+j_max)
                DatasetIdx[idx].append(len(Dataset))
                Dataset.append(templates[idx]/np.linalg.norm(templates[idx])*np.linalg.norm(Data[i+j_max-20:i+j_max+21]))
                Data[i+j_max-20:i+j_max+21] = Data[i+j_max-20:i+j_max+21]-templates[idx]/np.linalg.norm(templates[idx])*np.linalg.norm(Data[i+j_max-20:i+j_max+21])
            else:
                i = i+1
            #print(str(i)+': '+str(time.time()-t)) 
        
        for var in list(locals()):
            if var != 'frames' or var != 'Data ' or var != 'Data set' or var !='DatasetIdx':
                del locals()[var]
        import gc
        gc.collect()

        return frames, Data , Dataset, DatasetIdx
    
def CrossCorrelogram(f_cluster_1, f_cluster_2, SamplingRate, NumFrames):
    """
    Compute and inspect the cross-correlogram between two spike trains.

    The function converts spike-frame indices into Neo SpikeTrain objects,
    computes the cross-correlation histogram (CCG), and compares the average
    activity around zero lag with activity in the side regions to detect
    potential refractory-period violations.

    Parameters
    ----------
    f_cluster_1 : array-like
        Spike-frame indices for the first cluster.
    f_cluster_2 : array-like
        Spike-frame indices for the second cluster.
    SamplingRate : float
        Acquisition sampling frequency in Hz.
    NumFrames : int
        Total number of frames in the recording.

    Notes
    -----
    The function currently prints a qualitative assessment of refractory
    behavior rather than returning a value. A low central peak relative to
    the side regions may indicate over-clustering.
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

def ChannelSpksort(ch):
    """
    Retrieve the neighborhood channels around a reference channel.

    The function identifies all channels within a 5x5 spatial window centered
    on the specified channel and returns both the channel list and the
    position of the reference channel within that list.

    Parameters
    ----------
    ch : int
        Index of the reference channel on a 64-column electrode grid.

    Returns
    -------
    tuple
        Tuple containing:

        - chs : numpy.ndarray
            Sorted array of neighboring channel indices.
        - idx_ch : numpy.ndarray
            Index of the reference channel within ``chs``.
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

def LinkChsSpksort(results):
    """
    Estimate template overlap between neighboring channels.

    The function compares spike templates extracted from different channels
    using Pearson correlation and counts highly correlated template pairs.

    Parameters
    ----------
    results : list
        List containing spike sorting results for multiple channels.
        Each element is expected to contain the templates associated with
        a channel.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix where element (i, j) contains the number of template
        pairs with Pearson correlation greater than or equal to 0.95 between
        channels i and j.
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



