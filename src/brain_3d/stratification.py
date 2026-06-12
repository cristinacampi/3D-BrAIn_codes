"stratification functions"
import numpy as np
import warnings
np.warnings = warnings
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from scipy.cluster.hierarchy import dendrogram, linkage, fclusterdata
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA, FastICA, KernelPCA
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric
import pandas as pd
import math
import FCM
from scipy.signal import find_peaks, butter, filtfilt
import igraph as ig
import leidenalg as la
from kneed import KneeLocator
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

'''DISTANCES'''

def d_m(a, b, p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Minkowski distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        w_max (int): Unused. Defaults to 1.
        p_minkowski (int): parameter. Defaults to 2 (euclidean distance).
        g (int): Unused. Defaults to 1.
        epsilon_EDR (float): Unused. Defaults to 0.001.
        epsilon_LCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): Minkowski distance between a and b.
    """
    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        print('vectors with not equal length')
    else:
        d = minkowski(a, b, p=p_minkowski)
    
        return d
    
def MatrixC(M):
    """
    Creating the matrix C helpful in the dtw (and derivatives) distance.
    
    Args:
        M (np.ndarray): matrix of the distances between the entries of two vectors.
    
    Returns:
        C (np.ndarray): matrix helpful in the dtw (and derivatives) distance.
    """
    l1 = M.shape[0]
    C = np.zeros((l1+1, l1+1))
    C[:][:] = np.inf
    C[0][0] = 0
    for i in range(l1):
        i_C = i+1
        for j in range(l1):
            j_C = j+1
            C[i_C][j_C] = M[i][j] + min(C[i_C-1][j_C-1], C[i_C-1][j_C], C[i_C][j_C-1])
    return C

def Warping(a, b, M):
    """
    Realignment of a signal a on a signal b (or vice versa).
    
    Args:
        a (np.ndarray): array 1D.
        b (np.ndarray): array 1D.
        M (np.ndarray): 2D matrix containing the punctual ed distances between the entries of the vectors a and b.
    
    Returns:
        tuple: realignments and indexes of the realignment.
    """
    C = MatrixC(M)
    i = C.shape[0]-1
    j = C.shape[1]-1
    l = []
    while (i>0) & (j>0):
        l.append((i,j))
        m = min(C[i-1][j],C[i][j-1], C[i-1][j-1])
        if m == C[i-1][j-1]:
            i = i-1
            j = j-1
        elif m == C[i][j-1]:
            i = i
            j = j-1
        elif m == C[i-1][j]:
            i = i-1
            j = j
    idx_a = []
    idx_b = []
    for k in range(len(l)):
        idx_a.append(l[len(l)-1-k][0])
        idx_b.append(l[len(l)-1-k][1])

    x = range(len(a))
    w_a = np.zeros(len(idx_a))
    w_b = np.zeros(len(idx_a))
    for i in range(len(idx_a)):
        w_a[i] = a[idx_a[i]-1]
        w_b[i] = b[idx_b[i]-1]
    idx_a = np.array(idx_a)-1
    idx_b = np.array(idx_b)-1
 
    
    return w_a, w_b, idx_a, idx_b

def MatrixM(a, b):
    """
    2D matrix containing the punctual ed distances between the entries of the vectors a and b.
    
    Args:
        a (np.ndarray): vector of length n.
        b (np.ndarray): vector of length n.
    
    Returns:
        M (np.ndarray): 2D matrix containing the punctual ed distances between the entries of the vectors a and b.
    """
    l1 = len(a)
    '''
    M = np.zeros((l1, l1))
    for i in range(l1):
        for j in range(l1):
            M[i][j] = (a[i]-b[j])**2
    '''
    aa = np.repeat(a,l1,axis=0)
    aa = np.reshape(aa,(l1,l1))
    bb = np.repeat(b,l1,axis=0)
    bb = np.reshape(bb,(l1,l1))
    bb = bb.T
    M = (aa-bb)**2

    
    return M

def d_dtw(a, b, p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Dynamic Time Warping distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        p_minkowski (int): Unused. Defaults to 2.
        w_max (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilon_EDR (float): Unused. Defaults to 0.001.
        epsilon_LCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): dtw distance between a and b.
    """
    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        print('vectors with not equal length')
    else:
        M = MatrixM(a, b)
        C = MatrixC(M)
        d = C[l1][l1]
        return math.pow(d,0.5)

def a1_b1_ddtw(a, b):
    """
    The new vectors built for the ddtw (wddtw) distance of the original vectors.
    
    Args:
        a (np.ndarray): vector of length n.
        b (np.ndarray): vector of length n.
    
    Returns:
        tuple: the new vectors for the ddtw (wddtw) distance between a and b.
    """
    l1 = len(a)
    a1 = np.empty(l1-2)
    b1 = np.empty(l1-2)
    indexes =np.array(range(l1-2))+1
    for i in indexes:
        a1[i-1]=((a[i]-a[i-1])+((a[i+1]-a[i-1])/2))/2
        b1[i-1]=((b[i]-b[i-1])+((b[i+1]-b[i-1])/2))/2
    return a1, b1

def d_ddtw(a, b, p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Derivative Dynamic Time Warping distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        p_minkowski (int): Unused. Defaults to 2.
        w_max (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilon_EDR (float): Unused. Defaults to 0.001.
        epsilon_LCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): ddtw distance between a and b.
    """
    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        print('vectors with not equal length')
    else:
        a1, b1 = a1_b1_ddtw(a, b)
        d = d_dtw(a1,b1)
        return d
    
def d_wdtw(a, b, p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Weighted Dynamic Time Warping distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        p_minkowski (int): Unused. Defaults to 2.
        w_max (int): upper bound of the weights. Defaults to 1.
        g (int): exponential parameter. Defaults to 1.
        epsilon_EDR (float): Unused. Defaults to 0.001.
        epsilon_LCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): wdtw distance between a and b.
    """
    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        print('vectors with not equal length')
    else:
        #w_max = 1 #da mettere a scelta dell'utente
        #g = 1 #da mettere a scelta dell'utente
        M = MatrixM(a, b)
        M = MatrixMw(M, w_max, g)
        C = MatrixC(M)        
        d = C[l1][l1]
        return math.pow(d,0.5)
    
def MatrixMw(M, w_max, g):
    """
    Matrix of the punctual distances between the entries of two vectors but with weights based on the indexes of the entries.
    
    Args:
        M (np.ndarray): matrix of the ed distances between the entries of two vectors.
        w_max (float): parameter for the weigths.
        g (float): parameter for the weigths.
    
    Returns:
        Mw (np.ndarray): Matrix of the punctual distances between the entries of two vectors but with weights based on the indexes of the entries.
    """
    l1=M.shape[0]

    for i in range(l1):
            for j in range(l1):
                M[i,j] = (w_max/(1+math.exp(-g*(abs(i-j)-l1/2))))*M[i,j]
    return M
    
def d_wddtw(a, b, p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate = 1000):
    """
    Derivative Weigthed Dynamic Time Warping distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        p_minkowski (int): Unused. Defaults to 2.
        w_max (int): upper bound of the weights. Defaults to 1.
        g (int): exponential parameter. Defaults to 1.
        epsilon_EDR (float): Unused. Defaults to 0.001.
        epsilon_LCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): wdtw distance between a and b.
    """
    a1, b1 = a1_b1_ddtw(a, b)
    d = d_wdtw(a1, b1, w_max = w_max, g = g)
    return d

def d_lcss(a, b, p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Longest Common Subsequence distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        p_minkowski (int): Unused. Defaults to 2.
        w_max (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilon_EDR (float): Unused. Defaults to 0.001.
        epsilon_LCSS (float): Threshold. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): lcss distance between a and b.
    """
    epsilon_LCSS_abs = epsilon_LCSS*np.linalg.norm(a)

    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        print('vectors with not equal length')
    else:
        L = np.zeros((l1+1, l1+1))
        for i in range(l1):
            i_L = i+1
            for j in range(l1):
                j_L = j+1
                if abs(a[i]-b[j])<epsilon_LCSS_abs:
                    L[i_L][j_L] = L[i_L-1][j_L-1]+1
                else:
                    L[i_L][j_L] = max(L[i_L-1][j_L], L[i_L][j_L-1])
        LCSS = L[l1][l1]
        d = 1 - LCSS/l1
        return d

def d_edr(a, b, p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Edit distance on Real Sequences.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        p_minkowski (int): Unused. Defaults to 2.
        w_max (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilon_EDR (float): Threshold. Defaults to 0.001.
        epsilon_LCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): edr distance between a and b.
    """
    epsilon_EDR_abs = epsilon_EDR*np.linalg.norm(a)

    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        print('vectors with not equal length')
    else:
        E = np.zeros((l1+1, l1+1))

        for i in range(l1):
            i_E = i+1
            for j in range(l1):
                j_E = j+1
                if abs(a[i]-b[j])<epsilon_EDR_abs:
                    c = 0
                else:
                    c = 1
                match = E[i_E-1][j_E-1]+c
                insert = E[i_E-1][j_E]+1
                delete = E[i_E][j_E-1]+1
                E[i_E][j_E] = min(match, insert, delete)
        d = E[l1][l1]
        return d

def d_rho_2 (a, b, p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Distance based on the Pearson's correlation coefficient.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        p_minkowski (int): Unused. Defaults to 2.
        w_max (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilon_EDR (float): Unused. Defaults to 0.001.
        epsilon_LCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): rho2 distance between a and b.
    """

    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        print('vectors with not equal length')
    else:
        rho = pearsonr(a,b)[0]
        d = 2*(1-rho)
    return d

def d_sts(a, b, p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Short Time Series distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        p_minkowski (int): Unused. Defaults to 2.
        w_max (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilon_EDR (float): Unused. Defaults to 0.001.
        epsilon_LCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Sampling Frequency. Defaults to 1000.
    
    Returns:
        d (float): sts distance between a and b.
    """
    
    if len(a) != len(b):
        print('vectors with not equal length')
    else:
        aa = np.diff(a)
        bb = np.diff(b)
        aux = ((aa-bb)*SamplingRate)**2
        d = math.sqrt(np.sum(aux))
        return d
    
def AdjacencyMatrix(data, distance_str = 'm', p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Adjacency matrix for Leiden algorithm based on metric selected.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        distance_str (str): metric used to compute distances. Defaults to 'm'.
        p_minkowski (int): Minkowski parameter. Defaults to 2.
        w_max (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilon_EDR (float): EDR threshold. Defaults to 0.001.
        epsilon_LCSS (float): LCSS threshold. Defaults to 0.001.
        SamplingRate (int): STS parameter. Defaults to 1000.
    
    Returns:
        adjacency (np.ndarray): 2D matrix adjacency matrix for Leiden graph-based algorithm.
    """

    distance = 'd_'+distance_str
    distance = globals()[distance]
    distance.__defaults__ = (p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
    dim = data.shape[0]
    matrix = np.zeros((dim,dim))
    for i in range(dim):
        for j in np.array(range(i+1,dim)):
            matrix[i][j]=distance(data[i], data[j])
    matrix = matrix + matrix.T
    M = np.max(matrix)
    matrix_2 = matrix/M
    adjacency = 1/(1+matrix_2)
    adjacency[adjacency<=0.75]=0

    return adjacency

'''NORMALIZATION'''
def NormalizationMinMaxSingle(data):
    """
    Normalization of a dataset following the formula data[i] = (data[i]-m)/(M-m) where m and M are respectively the minimum.
    
    Args:
        data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the dataset.
    
    Returns:
        data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the dataset normalized.
    """
    for i in range(len(data)):
        m = min(data[i,:])
        M = max(data[i,:])
        if m == M :
            data[i,:] = data[i,:]-data[i,:]
        else:
            data[i,:] = (data[i,:]-m)/(M-m) 

    return data 

def NormalizationMinMaxGlobal(data):
    """
    Normalization of a dataset following the formula data[i] = (data[i]-m)/(M-m) where m and M are respectively the global minimum.
    
    Args:
        data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the dataset.
    
    Returns:
        data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the dataset normalized.
    """
   
    size = len(data)
    m = np.zeros(size)
    M = np.zeros(size)
    for i in range(size):
        m[i] = min(data[i,:])
        M[i] = max(data[i,:])
    minimum = min(m)
    maximum = max(M)
    for i in range(size):
        data[i,:] = (data[i,:]-minimum)/(maximum-minimum)

    return data

def Whitening(data):
    """
    Normalization of a dataset following the formula data[i] = (data[i]-mu)/sigma where mu and sigma are respectively the mean.
    
    Args:
        data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the dataset.
    
    Returns:
        data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the dataset normalized.
    """
    for i in  range(len(data)):
        m = min(data[i,:])
        M = max(data[i,:])
        mu = np.mean(data[i,:])
        sigma = np.std(data[i,:])
        if m == M:
            data[i,:] = data[i,:]-data[i,:]
        else:
            data[i,:] = (data[i,:]-mu)/sigma

    return data

def WhiteningGlobal(data):
    """
    Normalization of a dataset following the formula data[i] = (data[i]-mu)/sigma where mu and sigma are respectively.
    
    Args:
        data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the dataset.
    
    Returns:
        data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the dataset normalized.
    """
    size = len(data)
    mu = np.mean(data)
    sigma = np.std(data)
    for i in range(size):
        data[i,:] = (data[i,:]-mu)/sigma 

    return data   


'''ALGORITHMS'''

def Dendrogram(data, distance, method_HC ='complete', threshold_dendrogram=0.7):
    """
    Given a dataset, a metric, a method and a threshold, the function returns the dendrogram plot of the chosen hierarchical.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        distance (str): metric used to compute distances.
        method_HC (str): linkage method.
        threshold_dendrogram (float): threshold between 0 and 1 that multiplied with the height of the dendrogram represent the height at which we want to cut it.
    
    Returns:
        max_d (float): height of the cut of the dendrogram.
    """

    try:
        linkage_data = linkage(data, method = method_HC, metric = distance)
        n = len(data)
        aggregation_levels = linkage_data[:,2]
        max_d = threshold_dendrogram * aggregation_levels[n-2]
    except:
        max_d = threshold_dendrogram*len(data)
    '''
    plt.figure()
    plt.title("Dendrogram")
    dendrogram(linkage_data)
    plt.axhline(y = max_d, c='k')
    plt.show()
    #plt.savefig("Dendrogram.png")
    # '''
    return max_d

def HierarchicalClustering(data, method_HC, distance, threshold_dendrogram, max_classes, criterion, distance_str, p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate):
    """
    Given a dataset, a distance, a method and a threshold, the function returns the clusters built by the chosen hierarchical.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        method_HC (str): linkage method.
        distance (function): metric used to compute distances.
        threshold_dendrogram (float): height at which we cut the dendrogram to form clusters.
        max_classes (int): maximum number of classes to obtain from the clustering.
        criterion (str): clustering criterion.
        distance_str (str): metric used to compute distances.
        p_minkowski (int): Defaults to 2.
        w_max (int): upper bound of the weights. Defaults to 1.
        g (int): exponential parameter. Defaults to 1.
        epsilon_EDR (float): Defaults to 0.001.
        epsilon_LCSS (float): Defaults to 0.001.
        SamplingRate (float): Defaults to 1000.
    
    Returns:
        clusters (list): clusters of the applied algorithm.
    """
    if criterion == 'distance':
        fclust = fclusterdata(data, threshold_dendrogram, criterion = 'distance', metric = distance, method = method_HC)

    elif criterion=='maxclust':
        if type(max_classes) == int:
            k_elbow = max_classes
            fclust = fclusterdata(data, k_elbow, criterion = 'maxclust', metric = distance, method = method_HC) 
        else:
            score = -1000
            for k_elbow in max_classes:
                print(k_elbow)
                fclust_aux = fclusterdata(data, k_elbow, criterion = 'maxclust', metric = distance, method = method_HC) 
                clusters_aux = []
                for i in range(max(fclust_aux)):
                    indexes = np.where(fclust_aux==i+1)[0]
                    clusters_aux.append(indexes)
                clusters_aux = [max(fclust_aux), clusters_aux]
                score_aux  = Silhouette(data, clusters_aux, distance_str, p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
                if score_aux>score:
                    score = score_aux
                    fclust = fclust_aux
    clusters = []
    for i in range(max(fclust)):
        indexes = np.where(fclust==i+1)[0]
        clusters.append(indexes)
    
    return clusters


def Kshape_algo(data, nc2test, method_KM='silhouette'):
    """
    Given a dataset and the possible number of clusters to use, the function applies the k-shape algorithm.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset (n_samples x n_timesteps).
        nc2test (np.ndarray): a vector with the possible choices in term of the number of clusters.
        method_KM (str): method to compute the optimal number of clusters. Defaults to 'silhouette'.
    
    Returns:
        labels (np.ndarray): cluster labels for each sample.
        c_best (int): optimal number of clusters.
        centers (np.ndarray): coordinates of the centers of the clusters formed.
    """

    # KShape richiede dati normalizzati e shape (n_samples, n_timesteps, 1)
    data_scaled = TimeSeriesScalerMeanVariance().fit_transform(data)

    iterations = []
    best_labels = None
    best_centers = None
    c_best = nc2test[0]

    if method_KM == 'davies_bouldin':
        best_score = float('inf')
    else:
        best_score = -float('inf')

    if len(nc2test) == 1:
        n_clusters = nc2test[0]
        ks = KShape(n_clusters=n_clusters, random_state=42)
        labels = ks.fit_predict(data_scaled)
        return labels, n_clusters, ks.cluster_centers_

    else:
        for n_clusters in nc2test:
            ks = KShape(n_clusters=n_clusters, random_state=42)
            labels = ks.fit_predict(data_scaled)

            # WCSS con distanza shape-based non ha senso, usiamo l'inertia di KShape
            wcss = ks.inertia_

            if len(np.unique(labels)) > 1:
                # per silhouette usiamo i dati originali 2D (n_samples x n_timesteps)
                data_2d = data_scaled.reshape(data_scaled.shape[0], -1)
                if method_KM == 'silhouette':
                    score = silhouette_score(data_2d, labels, sample_size=1000, random_state=42)
                elif method_KM == 'davies_bouldin':
                    score = davies_bouldin_score(data_2d, labels)
                elif method_KM == 'calinski_harabasz':
                    score = calinski_harabasz_score(data_2d, labels)
                elif method_KM == 'wcss':
                    score = wcss
            else:
                score = 0

            if method_KM == 'davies_bouldin':
                if score < best_score:
                    best_score = score
                    c_best = n_clusters
                    best_labels = labels
                    best_centers = ks.cluster_centers_
            else:
                if score > best_score:
                    best_score = score
                    c_best = n_clusters
                    best_labels = labels
                    best_centers = ks.cluster_centers_

            print(f"For n_centroids = {n_clusters}, {method_KM} score is {score}")
            print(f"For n_centroids = {n_clusters}, davies_bouldin score is {davies_bouldin_score(data_2d, labels)}")
            print(f"For n_centroids = {n_clusters}, calinski_harabasz score is {calinski_harabasz_score(data_2d, labels)}")
            print(f"For n_centroids = {n_clusters}, wcss score is {wcss}")

            iterations.append(score)

        return best_labels, c_best, best_centers

def Kmeans_algo(data, nc2test, distance, method_KM = 'silhouette'):
    """
    Given a dataset, a metric and the possible number of centroids to use, the function applies the k-means algorithm.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        nc2test (np.ndarray): a vector with the possible choices in term of the number of centroids to use to apply the algorithm.
        distance (str): metric used to compute distances.
        method_KM (str): method to compute the optimal number of centroids. Defaults to 'silhouette'.
    """

  
    metric = distance_metric(type_metric.USER_DEFINED, func=distance)
    iterations = []
    classi = []
    centers = []
    c_best = 1
    if method_KM == 'davies_bouldin':
        best_score = float('inf')
    else:
        best_score = -float('inf')

    if len(nc2test) == 1:
        n_clusters = nc2test[0]
        start_centers = kmeans_plusplus_initializer(data, n_clusters).initialize();
        kmeans_instance = kmeans(data, start_centers, metric=metric)
        # run cluster analysis and obtain results
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        classi.append((n_clusters, clusters))
        centers.append(kmeans_instance.get_centers())
        return classi[0][1], n_clusters, centers[0] 
    
    else:
        for k in range(len(nc2test)):
            #random 
            n_clusters = nc2test[k] 
            n = len(data)
            start_centers = kmeans_plusplus_initializer(data, n_clusters).initialize();
            kmeans_instance = kmeans(data, start_centers, metric=metric)
            # run cluster analysis and obtain results
            kmeans_instance.process()
            clusters = kmeans_instance.get_clusters()
            n_clusters_postK = len(clusters)
            classi.append((n_clusters_postK, clusters))
            centers.append(kmeans_instance.get_centers())
            labels = np.array(range(n))
            wcss = 0
            for j in range(len(clusters)):
                labels[clusters[j]] = j+1
                cluster_points = data[clusters[j]] 
                for i in range(cluster_points.shape[0]): 
                    wcss += distance(cluster_points[i], centers[k][j])**2
            iterations.append(wcss)

            if len(np.unique(labels))>1:  
                if method_KM == 'silhouette': 
                    score = silhouette_score(data, labels, metric=metric, sample_size=1000, random_state=42)
                elif method_KM == 'davies_bouldin':
                    score = davies_bouldin_score(data, labels)
                elif method_KM == 'calinski_harabasz':
                    score = calinski_harabasz_score(data, labels)
                elif method_KM == 'wcss':
                    score = wcss
            else:
                score = 0 

            if method_KM == 'davies_bouldin':
                if score < best_score:  
                    best_score = score
                    c_best = n_clusters
            else:
                if score > best_score:
                    best_score = score
                    c_best = n_clusters
               
     
            print(f"For n_centroids = {n_clusters}, {method_KM} score is {score}")
            print(f"For n_centroids = {n_clusters}, davies_bouldin score is {davies_bouldin_score(data, labels)}")
            print(f"For n_centroids = {n_clusters}, calinski_harabasz score is {calinski_harabasz_score(data, labels)}")
            print(f"For n_centroids = {n_clusters}, wcss score is {wcss}")


            iterations.append(score) 
            
            
        if method_KM == 'wcss':
            nc2test_array = np.array(nc2test)
            kl = KneeLocator(nc2test_array, iterations, curve="convex", direction="decreasing")
            k_elbow = kl.elbow
            if k_elbow is None:
                k_elbow=nc2test_array[-1]
            idx_best = np.where(nc2test_array==k_elbow)[0][0]
        else:
            nc2test_array = np.array(nc2test)
            try:
                idx_best = np.where(nc2test_array==c_best)[0][0]
            except:
                idx_best = 0
    
        return classi[idx_best][1], classi[idx_best][0], centers[idx_best] 
    

def Silhouette(data, clusters, distance, p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate):
    """
    silhouette score of the clusters given the distance and the parameters of the distance.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        clusters (tuple): number of clusters and clusters obtained by the clustering algorithm applied.
        distance (str): metric used to compute distances.
        p_minkowski (int): Defaults to 2.
        w_max (int): upper bound of the weights. Defaults to 1.
        g (int): exponential parameter. Defaults to 1.
        epsilon_EDR (float): Defaults to 0.001.
        epsilon_LCSS (float): Defaults to 0.001.
        SamplingRate (float): Defaults to 1000.
    """
    labels = np.array(range(data.shape[0]))
    for j in range(clusters[0]):
        labels[clusters[1][j]] = j+1
    if len(np.unique(labels))>1:
        d = 'd_'+distance
        d = globals()[d]
        d.__defaults__ = (p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
        score = silhouette_score(data, labels, metric=d)
    else:
        score = 0 

    return score

def ICA_algo(data, ncomp = 10):
    """
    Apply Independent Component Analysis to reduce dimensionality and reconstruct the data.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset (n_samples x n_features).
        ncomp (int): maximum number of independent components to compute. Defaults to 10.
    
    Returns:
        X_transformed (np.ndarray): dataset projected on the selected independent components.
        X_back (np.ndarray): reconstructed dataset projected back to the original feature space.
    """
    n = min(data.shape[0], data.shape[1], ncomp)
    ICA = FastICA(n_components=n, random_state=0)
    X_transformed = ICA.fit_transform(data)
    X_back = ICA.inverse_transform(X_transformed)
    return X_transformed, X_back 

def kernelPCA_algo(data, ncomp = 10):
    """
    Given a dataset and a threshold between (0 and 1) in the term of the dispersion of the data to maintain, the function applies.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        ncomp (int): componentes number. Defaults to 10.
    
    Returns:
        X_transformed (np.ndarray): the dataset projected on the principal components selected from the kernel PCA.
        X_back (np.ndarray): the dataset projected back on the original dataset.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    n = min(data.shape[0], data.shape[1], ncomp)
    pca = KernelPCA(n_components=n, fit_inverse_transform=True)
    pca.fit(data_scaled)
    X_transformed = pca.fit_transform(data_scaled)
    X_back = pca.inverse_transform(X_transformed)
    X_back = scaler.inverse_transform(X_back)

    return X_transformed, X_back

def PCA_algo(data, threshold_variance = 0.9):
    """
    Given a dataset and a threshold between (0 and 1) in the term of the dispersion of the data to maintain, the function applies.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        threshold_variance (float): A number between 0 and 1 that represent the amount of the dispersion of data to maintain apllying the PCA. Defaults to 0.9.
    
    Returns:
        X_transformed (np.ndarray): the dataset projected on the principal components selected from the PCA.
        X_back (np.ndarray): the dataset projected back on the original dataset.
    """
    n = min(data.shape[0], data.shape[1])
    pca = PCA(n_components=n)
    pca.fit(data)
    variance = pca.explained_variance_ratio_
    sum_ratio = 0
    i = 0
    while sum_ratio < threshold_variance:
        sum_ratio += variance[i]
        i += 1
    n_c = i
    #print("Number of components to select: " +str(n_c))
    pca = PCA(n_components=n_c)
    X_transformed  = pca.fit_transform(data)
    X_back = pca.inverse_transform(X_transformed)

    return X_transformed, X_back

def Leiden_algo(data, threshold_Leiden=0.95, distance_str = 'm', p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Leiden graph based algorithm.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        threshold_Leiden (float): if graph based on Pearson's correlation coefficient we put to zero the weights of the edges under the threshold. Defaults to 0.95.
        distance_str (str): metric to compute the adjacency matrix. Defaults to 'm'.
        p_minkowski (int): Minkowski parameter. Defaults to 2.
        w_max (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilon_EDR (float): EDR threshold. Defaults to 0.001.
        epsilon_LCSS (float): LCSS threshold. Defaults to 0.001.
        SamplingRate (int): STS parameter. Defaults to 1000.
    
    Returns:
        clusters (list): clusters of the applied algorithm.
        G (ig graph): Leiden graph.
        partition (ig graph object): object containing the clusters labels.
    """
    #distance ='rho'
    if distance_str =='rho':
        df = pd.DataFrame(data)
        c = df.corr()
        c[c<=threshold_Leiden]=0

    else:
        c = AdjacencyMatrix(data.T, distance_str, p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate) 

    G =ig.Graph.Weighted_Adjacency(c, mode='undirected', attr='weight', loops=False)
    partition=la.find_partition(G, la.ModularityVertexPartition)
    optimiser = la.Optimiser()
    improvement = optimiser.optimise_partition(partition)
    while improvement:
        improvement = optimiser.optimise_partition(partition)
    partition_membership=partition.membership
    n_clusters = max(partition_membership)+1
    partition_membership = np.array(partition_membership)
    clusters =[[] for i in range(n_clusters)] 
    for i in range(n_clusters):
        idx = np.where(partition_membership==i)[0]
        clusters[i].append(idx) 

    return clusters, G, partition

def Clustering(data, algo = 'KM', distance_str = 'm', method_HC = 'complete', criterion_HC = 'distance', method_KM = 'silhouette', max_iter_FCM=10, threshold_variance = 0.9, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, fuzzy_parameter = 1, threshold_dendrogram = 0.7, max_classes = [2], threshold_Leiden = 0.9, SamplingRate = 1000, p_minkowski = 2, normalization = 'OFF', norm_mode ='min_max_single', ica_ncomp=10, kpca_ncomp=10): 
    """
    Given user choice, clustering algorithm.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        algo (str): clustering algorithm. Choices: c-means ('KM'), fuzzy c-means ('FCM'), hierarchical clustering ('HC'), Leiden ('Leiden') and applying first a dimensionality reduction by PCA ('PCA&KM', 'PCA&FCM', 'PCA&HC', 'PCA&Leiden'). Defaults to 'KM'.
        distance_str (str): metric used for clustering. Defaults to 'm'.
        method_HC (str): linkage method. Choices: 'complete', 'single', 'average'. Defaults to 'complete'.
        criterion_HC (str): Hierarchical clustering criterion. Choices: 'distance', 'maxclust'. Defaults to 'distance'.
        method_KM (str): Method to selct the optimal number of centroid. 'Choices: 'silhouette', 'wcss'. Defaults to 'silhouette'.
        max_iter_FCM (int): maximum number of iterations for FCM. Defaults to 10.
        threshold_variance (float): explained variance after PCA. Defaults to 0.9.
        w_max (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilon_EDR (float): EDR threshold. Defaults to 0.001.
        epsilon_LCSS (float): LCSS threshold. Defaults to 0.001.
        fuzzy_parameter (int): FCM parameter. Defaults to 1.
        threshold_dendrogram (float): Cut height in percentage. Defaults to 0.7.
        max_classes (list): Classes to test. Defaults to [2].
        threshold_Leiden (float): Leiden threshold. Defaults to 0.9.
        SamplingRate (int): STS parameter. Defaults to 1000.
        p_minkowski (int): Minkowski parameter. Defaults to 2.
        normalization (str): To applying normalization. Choices: 'ON', 'OFF'. Defaults to 'OFF'.
        norm_mode (str): If normalization applied, to select the modality. Choices: 'min_max_single', 'min_max_global', 'mu_std_single', 'mu_std_global'. Defaults to 'min_max_single'.
        ica_ncomp (int): the number of components for ICA.
        kpca_ncomp (int): the number of components for kernelPCA.
    """

    distance = 'd_'+distance_str
    distance = globals()[distance]
    distance.__defaults__ = (p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)

   
    n = data.shape[0]
    x = range(data.shape[1])
        
    data_plot = data.copy()

    # NORMALIZATION
    if normalization == 'ON':
        if norm_mode =='min_max_single':
            data = Whitening(data)
        elif norm_mode =='min_max_global':
            data = WhiteningGlobal(data)
        elif norm_mode =='mu_std_single':
            data = Whitening(data)
        else:
            data = WhiteningGlobal(data)


    if type(max_classes) == int:
        if max_classes < 2:
            max_classes = 2
        if max_classes >= len(data):
            max_classes = len(data)
        nc2test = [max_classes]
    else:
        if len(max_classes)==1:
            max_classes = max_classes[0] 
            if max_classes < 2:
                max_classes = 2
            if max_classes >= len(data):
                max_classes = len(data)
            nc2test = [max_classes]
        else: 
            nc2test = max_classes 

    if n==1:
        return (1, [[0]])
    else: 
        if algo == "HC":
            threshold_dendrogram = Dendrogram(data=data, method_HC=method_HC, distance=distance, threshold_dendrogram=threshold_dendrogram)
            clusters_HC = HierarchicalClustering(data=data, method_HC=method_HC, distance=distance, threshold_dendrogram=threshold_dendrogram, max_classes=nc2test, criterion = criterion_HC,
                                                 distance_str = distance_str,
                                                 p_minkowski = p_minkowski, w_max = w_max, g = g, epsilon_EDR = epsilon_EDR, epsilon_LCSS = epsilon_LCSS, SamplingRate = SamplingRate)
            n_classes = len(clusters_HC)
            clusters = []
            for i in range(n_classes):
                if len(clusters_HC[i])>0:
                    clusters.append(clusters_HC[i])
            n_classes = len(clusters)
        
        elif algo == "KM":  
            clusters, n_classes, centers = Kmeans_algo(data=data, nc2test=nc2test, distance=distance, method_KM=method_KM)

        
        elif algo == "FCM":
            clusters_KM, n_classes_KM, centers_KM = Kmeans_algo(data=data, nc2test=nc2test, distance=distance, method_KM=method_KM)
            clusters, centers, membership_mat = FCM.FCM(data=data, n_classes = n_classes_KM, centers=centers_KM, fuzzy_parameter=fuzzy_parameter, max_iter=max_iter_FCM, metric=distance)  
            n_classes = len(clusters)

        elif algo == "KShape":
            clusters, n_classes, centers = Kshape_algo(data=data, nc2test=nc2test, method_KM=method_KM)

        elif algo == "PCA&KShape":
            data, data_postPCA = PCA_algo(data=data, threshold_variance=threshold_variance)
            clusters, n_classes, centers = Kshape_algo(data=data, nc2test=nc2test, method_KM=method_KM)

        elif algo == "ICA&KShape":
            data, data_postICA = ICA_algo(data=data, ncomp = ica_ncomp)
            clusters, n_classes, centers = Kshape_algo(data=data, nc2test=nc2test, method_KM=method_KM)

        elif algo == "KernelPCA&KShape":
            data, data_postPCA = kernelPCA_algo(data=data, ncomp = kpca_ncomp)
            clusters, n_classes, centers = Kshape_algo(data=data, nc2test=nc2test, method_KM=method_KM)

    
    
        elif algo=="Leiden":
            clusters_L = Leiden_algo(data=data.T, threshold_Leiden=threshold_Leiden, distance_str=distance_str, p_minkowski = p_minkowski, w_max = w_max, g = g, epsilon_EDR = epsilon_EDR, epsilon_LCSS = epsilon_LCSS, SamplingRate=SamplingRate)[0]
            n_classes = len(clusters_L)
            clusters = []
            for i in range(n_classes):
                if len(clusters_L[i][0])>0:
                    clusters.append(clusters_L[i][0])
            n_classes = len(clusters)


        elif algo=="PCA&HC":
            data, data_postPCA  = PCA_algo(data=data, threshold_variance=threshold_variance)
            threshold_dendrogram = Dendrogram(data=data, method_HC=method_HC, distance=distance, threshold_dendrogram=threshold_dendrogram)
            clusters_HC = HierarchicalClustering(data=data, method_HC=method_HC, distance=distance, threshold_dendrogram=threshold_dendrogram, max_classes=nc2test, criterion=criterion_HC,
                                                 distance_str=distance_str,
                                                 p_minkowski = p_minkowski, w_max = w_max, g = g, epsilon_EDR = epsilon_EDR, epsilon_LCSS = epsilon_LCSS, SamplingRate = SamplingRate)
            n_classes = len(clusters_HC)
            
            clusters = []
            for i in range(n_classes):
                if len(clusters_HC[i])>0:
                    clusters.append(clusters_HC[i])
            n_classes = len(clusters)

        
        elif algo == "PCA&KM":
            data, data_postPCA = PCA_algo(data=data, threshold_variance=threshold_variance)
            clusters, n_classes, centers = Kmeans_algo(data=data, nc2test=nc2test, distance=distance, method_KM=method_KM)
          
            
        
        elif algo == "PCA&FCM":
            data, data_postPCA  = PCA_algo(data=data, threshold_variance=threshold_variance)
            clusters_KM, n_classes_KM, centers_KM = Kmeans_algo(data=data, nc2test=nc2test, distance=distance, method_KM=method_KM)
            max_iter = 5
            clusters, centers_FCM, membership_mat = FCM.FCM(data=data, n_classes = n_classes_KM, centers=centers_KM, fuzzy_parameter=fuzzy_parameter, max_iter=max_iter_FCM, metric=distance)
            n_classes = len(clusters)


        elif algo =="PCA&Leiden":
            data, data_postPCA  = PCA_algo(data=data, threshold_variance=threshold_variance)
            clusters_L = Leiden_algo(data=data.T, threshold_Leiden=threshold_Leiden, distance_str=distance_str, p_minkowski = p_minkowski, w_max = w_max, g = g, epsilon_EDR = epsilon_EDR, epsilon_LCSS = epsilon_LCSS, SamplingRate=SamplingRate)[0]
            n_classes = len(clusters_L)
            clusters = []
            for i in range(n_classes):
                if len(clusters_L[i][0])>0:
                    clusters.append(clusters_L[i][0])
            n_classes = len(clusters)

        elif algo=="ICA&HC":
            data, data_postICA = ICA_algo(data=data, ncomp = ica_ncomp)
            threshold_dendrogram = Dendrogram(data=data, method_HC=method_HC, distance=distance, threshold_dendrogram=threshold_dendrogram)
            clusters_HC = HierarchicalClustering(data=data, method_HC=method_HC, distance=distance, threshold_dendrogram=threshold_dendrogram, max_classes=nc2test, criterion=criterion_HC,
                                                 distance_str = distance_str,
                                                 p_minkowski = p_minkowski, w_max = w_max, g = g, epsilon_EDR = epsilon_EDR, epsilon_LCSS = epsilon_LCSS, SamplingRate = SamplingRate)
            n_classes = len(clusters_HC)
            clusters = []
            for i in range(n_classes):
                if len(clusters_HC[i])>0:
                    clusters.append(clusters_HC[i])
            n_classes = len(clusters)

        
        elif algo == "ICA&KM":
            data, data_postICA  = ICA_algo(data=data, ncomp = ica_ncomp)
            clusters, n_classes, centers = Kmeans_algo(data=data, nc2test=nc2test, distance=distance, method_KM=method_KM)

            
        elif algo == "ICA&FCM":
            data, data_postICA   = ICA_algo(data=data, ncomp = ica_ncomp)
            clusters_KM, n_classes_KM, centers_KM = Kmeans_algo(data=data, nc2test=nc2test, distance=distance, method_KM=method_KM)
            max_iter = 5
            clusters, centers_FCM, membership_mat = FCM.FCM(data=data, n_classes = n_classes_KM, centers=centers_KM, fuzzy_parameter=fuzzy_parameter, max_iter=max_iter_FCM, metric=distance)
            n_classes = len(clusters)


        elif algo =="ICA&Leiden":
            data, data_postICA   = ICA_algo(data=data, ncomp = ica_ncomp)
            clusters_L = Leiden_algo(data=data.T, threshold_Leiden=threshold_Leiden, distance_str=distance_str, p_minkowski = p_minkowski, w_max = w_max, g = g, epsilon_EDR = epsilon_EDR, epsilon_LCSS = epsilon_LCSS, SamplingRate=SamplingRate)[0]
            n_classes = len(clusters_L)
            clusters = []
            for i in range(n_classes):
                if len(clusters_L[i][0])>0:
                    clusters.append(clusters_L[i][0])
            n_classes = len(clusters)
            
        elif algo=="kernelPCA&HC":
            data, data_postPCA  = kernelPCA_algo(data=data, ncomp = kpca_ncomp)
            threshold_dendrogram = Dendrogram(data=data, method_HC=method_HC, distance=distance, threshold_dendrogram=threshold_dendrogram)
            clusters_HC = HierarchicalClustering(data=data, method_HC=method_HC, distance=distance, threshold_dendrogram=threshold_dendrogram, max_classes=nc2test, criterion=criterion_HC,
                                                 distance_str = distance_str,
                                                 p_minkowski = p_minkowski, w_max = w_max, g = g, epsilon_EDR = epsilon_EDR, epsilon_LCSS = epsilon_LCSS, SamplingRate = SamplingRate)
            n_classes = len(clusters_HC)
            clusters = []
            for i in range(n_classes):
                if len(clusters_HC[i])>0:
                    clusters.append(clusters_HC[i])
            n_classes = len(clusters)

        
        elif algo == "kernelPCA&KM":
            data, data_postPCA = kernelPCA_algo(data=data, ncomp = kpca_ncomp)
            clusters, n_classes, centers = Kmeans_algo(data=data, nc2test=nc2test, distance=distance, method_KM=method_KM)
          
            
        
        elif algo == "kernelPCA&FCM":
            data, data_postPCA  = kernelPCA_algo(data=data, ncomp = kpca_ncomp)
            clusters_KM, n_classes_KM, centers_KM = Kmeans_algo(data=data, nc2test=nc2test, distance=distance, method_KM=method_KM)
            max_iter = 5
            clusters, centers_FCM, membership_mat = FCM.FCM(data=data, n_classes = n_classes_KM, centers=centers_KM, fuzzy_parameter=fuzzy_parameter, max_iter=max_iter_FCM, metric=distance)
            n_classes = len(clusters)

        return (n_classes, clusters)

def RecursiveClustering(data, algo = 'KM', distance_str = 'm', method_HC = 'complete', criterion_HC = 'distance', method_KM = 'silhouette', max_iter_FCM=10, threshold_variance = 0.9, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, fuzzy_parameter = 1, noise = 0, threshold_dendrogram = 0.33, max_classes = [2], threshold_Leiden = 0.9, SamplingRate = 1000, p_minkowski = 2, normalization = 'OFF', norm_mode ='min_max_single'):
    """
    This algorithm is recursive, based on the sum of squares criteria.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        algo (str): clustering algorithm. Choices: c-means ('KM'), fuzzy c-means ('FCM'), hierarchical clustering ('HC'), Leiden ('Leiden') and applying first a dimensionality reduction by PCA ('PCA&KM', 'PCA&FCM', 'PCA&HC', 'PCA&Leiden'). Defaults to 'KM'.
        distance_str (str): metric used for clustering. Defaults to 'm'.
        method_HC (str): linkage method. Choices: 'complete', 'single', 'average'. Defaults to 'complete'.
        criterion_HC (str): Hierarchical clustering criterion. Choices: 'distance', 'maxclust'. Defaults to 'distance'.
        method_KM (str): Method to selct the optimal number of centroid. 'Choices: 'silhouette', 'wcss'. Defaults to 'silhouette'.
        max_iter_FCM (int): maximum number of iterations for FCM. Defaults to 10.
        threshold_variance (float): explained variance after PCA. Defaults to 0.9.
        w_max (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilon_EDR (float): EDR threshold. Defaults to 0.001.
        epsilon_LCSS (float): LCSS threshold. Defaults to 0.001.
        fuzzy_parameter (int): FCM parameter. Defaults to 1.
        noise (int): Percentage of noise to add to data. Defaults to 0.
        threshold_dendrogram (float): Cut height in percentage. Defaults to 0.7.
        max_classes (int): maximum number of classes. Defaults to 1.
        threshold_Leiden (float): Leiden threshold. Defaults to 0.9.
        SamplingRate (int): STS parameter. Defaults to 1000.
        p_minkowski (int): Minkowski parameter. Defaults to 2.
        normalization (str): To applying normalization. Choices: 'ON', 'OFF'. Defaults to 'OFF'.
        norm_mode (str): If normalization applied, to select the modality. Choices: 'min_max_single', 'min_max_global', 'mu_std_single', 'mu_std_global'. Defaults to 'min_max_single'.
    """
    
    distance = 'd_'+distance_str
    distance = globals()[d]
        
    data_plot = data.copy()

    distance.__defaults__ = (p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
    media = np.mean(data_plot,0)
    SS = 0
    for i in range(data_plot.shape[0]):
        SS += d(data_plot[i], media)**2
    clusters = Clustering(data=data, algo=algo, distance_str=distance_str, method_HC = method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, SamplingRate=SamplingRate, p_minkowski=p_minkowski, normalization=normalization, norm_mode=norm_mode) 
    clusters = clusters[1]    
    wcssk_list = [] 
    wcss_value = 0
    for j in range(len(clusters)):
        wcssk = 0
        cluster_points = data_plot[clusters[j]] 
        cluster_points=cluster_points.reshape((cluster_points.shape[-2],cluster_points.shape[-1]))
        center = np.mean(cluster_points,0)
        for i in range(cluster_points.shape[0]): 
            wcssk += d(cluster_points[i], center)**2
        wcss_value+=wcssk
        wcssk_list.append(wcssk)
    while max(wcssk_list)>SS/100*25:
        new_clusters = [] 
        for j in range(len(clusters)):
            if wcssk_list[j]>SS/100*25:
                data = data_plot.copy()
                clusters_j = Clustering(data=data[clusters[j]].reshape(data[clusters[j]].shape[-2],data[clusters[j]].shape[-1]), algo=algo, distance_str=distance_str, method_HC = method_HC, criterion_HC=criterion_HC, method_KM=method_KM, max_iter_FCM=max_iter_FCM, threshold_variance=threshold_variance, w_max=w_max, g=g, epsilon_EDR=epsilon_EDR, epsilon_LCSS=epsilon_LCSS, fuzzy_parameter=fuzzy_parameter, noise=noise, threshold_dendrogram=threshold_dendrogram, max_classes=max_classes, threshold_Leiden=threshold_Leiden, SamplingRate=SamplingRate, p_minkowski=p_minkowski, normalization=normalization, norm_mode=norm_mode)
                clusters_j = clusters_j[1] 
                indexes = np.array(sorted(clusters[j]))
                for i in range(len(clusters_j)):
                    new_clusters.append(list(indexes[clusters_j[i]]))
            else:
                new_clusters.append(clusters[j])
        clusters = new_clusters
        wcssk_list = [] 
        wcss_value = 0
        for j in range(len(clusters)):
            wcssk = 0
            cluster_points = data_plot[clusters[j]] 
            cluster_points=cluster_points.reshape((cluster_points.shape[-2],cluster_points.shape[-1]))
            center = np.mean(cluster_points,0)
            for i in range(cluster_points.shape[0]): 
                wcssk += distance(cluster_points[i], center)**2
            wcss_value+=wcssk
            wcssk_list.append(wcssk)
    
    return (len(clusters),clusters)
    
def ClusterCentroids(data, clusters):
    """
    Given a dataset and its subdivision in clusters, the function returns the centers of the clusters.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        clusters (tuple): number of clusters and clusters obtained by the clustering algorithm applied.
    
    Returns:
        centroids (np.ndarray): a vector with the coordinates of the centers of the clusters.
    """
    n_classes = len(clusters[1])
    centroids = np.zeros((n_classes, data.shape[1]))
    for i in range(n_classes):
        centroids[i]=np.mean(data[clusters[1][i]],0)
    return centroids 

def Classification(centroids, data, distance_str='m', p_minkowski = 2, w_max = 1, g = 1, epsilon_EDR = 0.001, epsilon_LCSS = 0.001, SamplingRate=1000):
    """
    Given a dataset, a group of centroids and a metric to compute distances between each data and each centroids,.
    
    Args:
        centroids (np.ndarray): a vector with the coordinates of the centers of some clusters.
        data (np.ndarray): 2D matrix representing the dataset.
        distance_str (str): metric to compute distances. Defaults to 'ed'.
    
    Returns:
        classification (list): clusters obtained from the dataset assigning each data to the class of the closest centroid.
    """
    distance = 'd_'+distance_str
    metric = globals()[distance]
    metric.__defaults__ = (p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
    n = data.shape[0]
    x = data.shape[1] 
    c = centroids.shape[0] 
    classification = []  
    for i in range(n):
        distances = [] 
        for j in range(c):
            distances.append(metric(data[i],centroids[j]))
        distances = np.array(distances)
        idx_m = np.where(distances==min(distances))[0][0] 
        idx_m = int(idx_m)
        classification.append(idx_m)

    return classification

def GaussianNoise(data, noise, seed):
    """
    Function to add Gaussian noise to data.
    
    Args:
        data (np.ndarray): 2D matrix representing the dataset.
        noise (float): amount of noise to add.
        seed (int): seed for random generation.
    
    Returns:
        data (np.ndarray): 2D matrix representing the dataset after the adding of noise.
    """
    np.random.seed(seed)
    for i in range(len(data)):
        sigma = np.std(data[i])
        if sigma == 0:
            data[i] = data[i]+noise*np.random.normal(0,1,len(data[i]))
        else:
            data[i] = data[i]+noise*np.random.normal(0,sigma,len(data[i]))

    return data


        