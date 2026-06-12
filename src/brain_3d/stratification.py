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
from . import FCM
import igraph as ig
import leidenalg as la
from kneed import KneeLocator
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

'''DISTANCES'''

def d_m(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Minkowski distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        wMax (int): Unused. Defaults to 1.
        pMinkowski (int): parameter. Defaults to 2 (euclidean distance).
        g (int): Unused. Defaults to 1.
        epsilonEDR (float): Unused. Defaults to 0.001.
        epsilonLCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): Minkowski distance between a and b.
    """
    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        print('vectors with not equal length')
    else:
        d = minkowski(a, b, p=pMinkowski )
    
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
        iC = i+1
        for j in range(l1):
            jC = j+1
            C[iC][jC] = M[i][j] + min(C[iC-1][jC-1], C[iC-1][jC], C[iC][jC-1])
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

def d_dtw(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Dynamic Time Warping distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        pMinkowski (int): Unused. Defaults to 2.
        wMax (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilonEDR (float): Unused. Defaults to 0.001.
        epsilonLCSS (float): Unused. Defaults to 0.001.
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

def d_ddtw(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Derivative Dynamic Time Warping distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        pMinkowski (int): Unused. Defaults to 2.
        wMax (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilonEDR (float): Unused. Defaults to 0.001.
        epsilonLCSS (float): Unused. Defaults to 0.001.
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
    
def d_wdtw(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Weighted Dynamic Time Warping distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        pMinkowski (int): Unused. Defaults to 2.
        wMax (int): upper bound of the weights. Defaults to 1.
        g (int): exponential parameter. Defaults to 1.
        epsilonEDR (float): Unused. Defaults to 0.001.
        epsilonLCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): wdtw distance between a and b.
    """
    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        print('vectors with not equal length')
    else:
        M = MatrixM(a, b)
        M = MatrixMw(M, wMax, g)
        C = MatrixC(M)        
        d = C[l1][l1]
        return math.pow(d,0.5)
    
def MatrixMw(M, wMax, g):
    """
    Matrix of the punctual distances between the entries of two vectors but with weights based on the indexes of the entries.
    
    Args:
        M (np.ndarray): matrix of the ed distances between the entries of two vectors.
        wMax (float): parameter for the weights.
        g (float): parameter for the weights.
    
    Returns:
        Mw (np.ndarray): Matrix of the punctual distances between the entries of two vectors but with weights based on the indexes of the entries.
    """
    l1=M.shape[0]

    for i in range(l1):
            for j in range(l1):
                M[i,j] = (wMax/(1+math.exp(-g*(abs(i-j)-l1/2))))*M[i,j]
    return M
    
def d_wddtw(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate = 1000):
    """
    Derivative Weigthed Dynamic Time Warping distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        pMinkowski (int): Unused. Defaults to 2.
        wMax (int): upper bound of the weights. Defaults to 1.
        g (int): exponential parameter. Defaults to 1.
        epsilonEDR (float): Unused. Defaults to 0.001.
        epsilonLCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): wdtw distance between a and b.
    """
    a1, b1 = a1_b1_ddtw(a, b)
    d = d_wdtw(a1, b1, wMax = wMax, g = g)
    return d

def d_lcss(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Longest Common Subsequence distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        pMinkowski (int): Unused. Defaults to 2.
        wMax (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilonEDR (float): Unused. Defaults to 0.001.
        epsilonLCSS (float): Threshold. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): lcss distance between a and b.
    """
    epsilonLCSS_abs = epsilonLCSS*np.linalg.norm(a)

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
                if abs(a[i]-b[j])<epsilonLCSS_abs:
                    L[i_L][j_L] = L[i_L-1][j_L-1]+1
                else:
                    L[i_L][j_L] = max(L[i_L-1][j_L], L[i_L][j_L-1])
        LCSS = L[l1][l1]
        d = 1 - LCSS/l1
        return d

def d_edr(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Edit distance on Real Sequences.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        pMinkowski (int): Unused. Defaults to 2.
        wMax (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilonEDR (float): Threshold. Defaults to 0.001.
        epsilonLCSS (float): Unused. Defaults to 0.001.
        SamplingRate (float): Unused. Defaults to 1000.
    
    Returns:
        d (float): edr distance between a and b.
    """
    epsilonEDR_abs = epsilonEDR*np.linalg.norm(a)

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
                if abs(a[i]-b[j])<epsilonEDR_abs:
                    c = 0
                else:
                    c = 1
                match = E[i_E-1][j_E-1]+c
                insert = E[i_E-1][j_E]+1
                delete = E[i_E][j_E-1]+1
                E[i_E][j_E] = min(match, insert, delete)
        d = E[l1][l1]
        return d

def d_rho_2 (a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Distance based on the Pearson's correlation coefficient.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        pMinkowski (int): Unused. Defaults to 2.
        wMax (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilonEDR (float): Unused. Defaults to 0.001.
        epsilonLCSS (float): Unused. Defaults to 0.001.
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

def d_sts(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Short Time Series distance.
    
    Args:
        a (np.ndarray): first time series.
        b (np.ndarray): second time series.
        pMinkowski (int): Unused. Defaults to 2.
        wMax (int): Unused. Defaults to 1.
        g (int): Unused. Defaults to 1.
        epsilonEDR (float): Unused. Defaults to 0.001.
        epsilonLCSS (float): Unused. Defaults to 0.001.
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
    
def AdjacencyMatrix(Data, DistanceStr = 'm', pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Adjacency matrix for Leiden Algorithm based on metric selected.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        DistanceStr (str): metric used to compute distances. Defaults to 'm'.
        pMinkowski (int): Minkowski parameter. Defaults to 2.
        wMax (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilonEDR (float): EDR threshold. Defaults to 0.001.
        epsilonLCSS (float): LCSS threshold. Defaults to 0.001.
        SamplingRate (int): STS parameter. Defaults to 1000.
    
    Returns:
        adjacency (np.ndarray): 2D matrix adjacency matrix for Leiden graph-based Algorithm.
    """

    distance = 'd_'+DistanceStr
    distance = globals()[distance]
    distance.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
    dim = Data.shape[0]
    matrix = np.zeros((dim,dim))
    for i in range(dim):
        for j in np.array(range(i+1,dim)):
            matrix[i][j]=distance(Data[i], Data[j])
    matrix = matrix + matrix.T
    M = np.max(matrix)
    matrix_2 = matrix/M
    adjacency = 1/(1+matrix_2)
    adjacency[adjacency<=0.75]=0

    return adjacency

'''Normalization'''
def NormalizationMinMaxSingle(Data):
    """
    Normalization of a Dataset following the formula Data[i] = (Data[i]-m)/(M-m) where m and M are respectively the minimum.
    
    Args:
        Data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the Dataset.
    
    Returns:
        Data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the Dataset normalized.
    """
    for i in range(len(Data)):
        m = min(Data[i,:])
        M = max(Data[i,:])
        if m == M :
            Data[i,:] = Data[i,:]-Data[i,:]
        else:
            Data[i,:] = (Data[i,:]-m)/(M-m) 

    return Data 

def NormalizationMinMaxGlobal(Data):
    """
    Normalization of a Dataset following the formula Data[i] = (Data[i]-m)/(M-m) where m and M are respectively the global minimum.
    
    Args:
        Data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the Dataset.
    
    Returns:
        Data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the Dataset normalized.
    """
   
    size = len(Data)
    m = np.zeros(size)
    M = np.zeros(size)
    for i in range(size):
        m[i] = min(Data[i,:])
        M[i] = max(Data[i,:])
    minimum = min(m)
    maximum = max(M)
    for i in range(size):
        Data[i,:] = (Data[i,:]-minimum)/(maximum-minimum)

    return Data

def Whitening(Data):
    """
    Normalization of a Dataset following the formula Data[i] = (Data[i]-mu)/sigma where mu and sigma are respectively the mean.
    
    Args:
        Data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the Dataset.
    
    Returns:
        Data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the Dataset normalized.
    """
    for i in  range(len(Data)):
        m = min(Data[i,:])
        M = max(Data[i,:])
        mu = np.mean(Data[i,:])
        sigma = np.std(Data[i,:])
        if m == M:
            Data[i,:] = Data[i,:]-Data[i,:]
        else:
            Data[i,:] = (Data[i,:]-mu)/sigma

    return Data

def WhiteningGlobal(Data):
    """
    Normalization of a Dataset following the formula Data[i] = (Data[i]-mu)/sigma where mu and sigma are respectively.
    
    Args:
        Data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the Dataset.
    
    Returns:
        Data (np.ndarray): 2D matrix (Nobs x Ntimes) representing the Dataset normalized.
    """
    size = len(Data)
    mu = np.mean(Data)
    sigma = np.std(Data)
    for i in range(size):
        Data[i,:] = (Data[i,:]-mu)/sigma 

    return Data   


'''AlgoRITHMS'''

def Dendrogram(Data, Distance, methodHC ='complete', ThresholdDendrogram=0.7):
    """
    Given a Dataset, a metric, a method and a threshold, the function returns the dendrogram plot of the chosen hierarchical.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        Distance (str): metric used to compute Distances.
        methodHC (str): linkage method.
        ThresholdDendrogram (float): threshold between 0 and 1 that multiplied with the height of the dendrogram represent the height at which we want to cut it.
    
    Returns:
        max_d (float): height of the cut of the dendrogram.
    """

    try:
        LinkageData = linkage(Data, method = methodHC, metric = Distance)
        n = len(Data)
        AggregationLevels = LinkageData[:,2]
        max_d = ThresholdDendrogram * AggregationLevels[n-2]
    except:
        max_d = ThresholdDendrogram*len(Data)
    '''
    plt.figure()
    plt.title("Dendrogram")
    dendrogram(LinkageData)
    plt.axhline(y = max_d, c='k')
    plt.show()
    #plt.savefig("Dendrogram.png")
    # '''
    return max_d

def HierarchicalClustering(Data, methodHC, Distance, ThresholdDendrogram, MaxClasses, criterion, DistanceStr, pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate):
    """
    Given a Dataset, a distance, a method and a threshold, the function returns the Clusters built by the chosen hierarchical.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        methodHC (str): linkage method.
        Distance (function): metric used to compute distances.
        ThresholdDendrogram (float): height at which we cut the dendrogram to form Clusters.
        MaxClasses (int): maximum number of classes to obtain from the clustering.
        criterion (str): clustering criterion.
        DistanceStr (str): metric used to compute distances.
        pMinkowski (int): Defaults to 2.
        wMax (int): upper bound of the weights. Defaults to 1.
        g (int): exponential parameter. Defaults to 1.
        epsilonEDR (float): Defaults to 0.001.
        epsilonLCSS (float): Defaults to 0.001.
        SamplingRate (float): Defaults to 1000.
    
    Returns:
        Clusters (list): Clusters of the applied Algorithm.
    """
    if criterion == 'distance':
        Fclust = fclusterdata(Data, ThresholdDendrogram, criterion = 'distance', metric = Distance, method = methodHC)

    elif criterion=='maxclust':
        if type(MaxClasses) == int:
            kElbow = MaxClasses
            Fclust = fclusterdata(Data, kElbow, criterion = 'maxclust', metric = Distance, method = methodHC)
        else:
            score = -1000
            for kElbow in MaxClasses:
                print(kElbow)
                FclustAux = fclusterdata(Data, kElbow, criterion = 'maxclust', metric = Distance, method = methodHC)
                ClustersAux = []
                for i in range(max(FclustAux)):
                    indexes = np.where(FclustAux==i+1)[0]
                    ClustersAux.append(indexes)
                ClustersAux = [max(FclustAux), ClustersAux]
                ScoreAux  = Silhouette(Data, ClustersAux, DistanceStr, pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
                if ScoreAux>score:
                    score = ScoreAux
                    Fclust = FclustAux
    Clusters = []
    for i in range(max(Fclust)):
        indexes = np.where(Fclust==i+1)[0]
        Clusters.append(indexes)
    
    return Clusters


def Kshape_Algo(Data, nc2test, methodKM='silhouette'):
    """
    Given a Dataset and the possible number of Clusters to use, the function applies the k-shape Algorithm.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset (n_samples x n_timesteps).
        nc2test (np.ndarray): a vector with the possible choices in term of the number of Clusters.
        methodKM (str): method to compute the optimal number of Clusters. Defaults to 'silhouette'.
    
    Returns:
        labels (np.ndarray): cluster labels for each sample.
        cBest (int): optimal number of Clusters.
        Centers (np.ndarray): coordinates of the Centers of the Clusters formed.
    """

    # KShape richiede dati normalizzati e shape (n_samples, n_timesteps, 1)
    DataScaled = TimeSeriesScalerMeanVariance().fit_transform(Data)

    iterations = []
    best_labels = None
    best_Centers = None
    cBest = nc2test[0]

    if methodKM == 'davies_bouldin':
        BestScore = float('inf')
    else:
        BestScore = -float('inf')

    if len(nc2test) == 1:
        nClusters = nc2test[0]
        ks = KShape(nClusters=nClusters, random_state=42)
        labels = ks.fit_predict(DataScaled)
        return labels, nClusters, ks.cluster_Centers_

    else:
        for nClusters in nc2test:
            ks = KShape(nClusters=nClusters, random_state=42)
            labels = ks.fit_predict(DataScaled)

            # WCSS con distanza shape-based non ha senso, usiamo l'inertia di KShape
            wcss = ks.inertia_

            if len(np.unique(labels)) > 1:
                # per silhouette usiamo i dati originali 2D (n_samples x n_timesteps)
                Data2d = DataScaled.reshape(DataScaled.shape[0], -1)
                if methodKM == 'silhouette':
                    score = silhouette_score(Data2d, labels, sample_size=1000, random_state=42)
                elif methodKM == 'davies_bouldin':
                    score = davies_bouldin_score(Data2d, labels)
                elif methodKM == 'calinski_harabasz':
                    score = calinski_harabasz_score(Data2d, labels)
                elif methodKM == 'wcss':
                    score = wcss
            else:
                score = 0

            if methodKM == 'davies_bouldin':
                if score < BestScore:
                    BestScore = score
                    cBest = nClusters
                    best_labels = labels
                    best_Centers = ks.cluster_Centers_
            else:
                if score > BestScore:
                    BestScore = score
                    cBest = nClusters
                    best_labels = labels
                    best_Centers = ks.cluster_Centers_

            print(f"For n_Centroids = {nClusters}, {methodKM} score is {score}")
            print(f"For n_Centroids = {nClusters}, davies_bouldin score is {davies_bouldin_score(Data2d, labels)}")
            print(f"For n_Centroids = {nClusters}, calinski_harabasz score is {calinski_harabasz_score(Data2d, labels)}")
            print(f"For n_Centroids = {nClusters}, wcss score is {wcss}")

            iterations.append(score)

        return best_labels, cBest, best_Centers

def Kmeans_Algo(Data, nc2test, distance, methodKM = 'silhouette'):
    """
    Given a Dataset, a metric and the possible number of Centroids to use, the function applies the k-means Algorithm.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        nc2test (np.ndarray): a vector with the possible choices in term of the number of Centroids to use to apply the Algorithm.
        distance (str): metric used to compute distances.
        methodKM (str): method to compute the optimal number of Centroids. Defaults to 'silhouette'.
    """

  
    metric = distance_metric(type_metric.USER_DEFINED, func=distance)
    iterations = []
    classi = []
    Centers = []
    cBest = 1
    if methodKM == 'davies_bouldin':
        BestScore = float('inf')
    else:
        BestScore = -float('inf')

    if len(nc2test) == 1:
        nClusters = nc2test[0]
        StartCenters = kmeans_plusplus_initializer(Data, nClusters).initialize();
        KmeansInstance = kmeans(Data, StartCenters, metric=metric)
        # run cluster analysis and obtain results
        KmeansInstance.process()
        Clusters = KmeansInstance.get_Clusters()
        classi.append((nClusters, Clusters))
        Centers.append(KmeansInstance.get_Centers())
        return classi[0][1], nClusters, Centers[0] 
    
    else:
        for k in range(len(nc2test)):
            #random 
            nClusters = nc2test[k] 
            n = len(Data)
            StartCenters = kmeans_plusplus_initializer(Data, nClusters).initialize();
            KmeansInstance = kmeans(Data, StartCenters, metric=metric)
            # run cluster analysis and obtain results
            KmeansInstance.process()
            Clusters = KmeansInstance.get_Clusters()
            nClusters_postK = len(Clusters)
            classi.append((nClusters_postK, Clusters))
            Centers.append(KmeansInstance.get_Centers())
            labels = np.array(range(n))
            wcss = 0
            for j in range(len(Clusters)):
                labels[Clusters[j]] = j+1
                cluster_points = Data[Clusters[j]] 
                for i in range(cluster_points.shape[0]): 
                    wcss += distance(cluster_points[i], Centers[k][j])**2
            iterations.append(wcss)

            if len(np.unique(labels))>1:  
                if methodKM == 'silhouette': 
                    score = silhouette_score(Data, labels, metric=metric, sample_size=1000, random_state=42)
                elif methodKM == 'davies_bouldin':
                    score = davies_bouldin_score(Data, labels)
                elif methodKM == 'calinski_harabasz':
                    score = calinski_harabasz_score(Data, labels)
                elif methodKM == 'wcss':
                    score = wcss
            else:
                score = 0 

            if methodKM == 'davies_bouldin':
                if score < BestScore:  
                    BestScore = score
                    cBest = nClusters
            else:
                if score > BestScore:
                    BestScore = score
                    cBest = nClusters
               
            print(f"For n_Centroids = {nClusters}, {methodKM} score is {score}")
            print(f"For n_Centroids = {nClusters}, davies_bouldin score is {davies_bouldin_score(Data, labels)}")
            print(f"For n_Centroids = {nClusters}, calinski_harabasz score is {calinski_harabasz_score(Data, labels)}")
            print(f"For n_Centroids = {nClusters}, wcss score is {wcss}")


            iterations.append(score) 
            
            
        if methodKM == 'wcss':
            nc2test_array = np.array(nc2test)
            kl = KneeLocator(nc2test_array, iterations, curve="convex", direction="decreasing")
            kElbow = kl.elbow
            if kElbow is None:
                kElbow=nc2test_array[-1]
            IdxBest = np.where(nc2test_array==kElbow)[0][0]
        else:
            nc2test_array = np.array(nc2test)
            try:
                IdxBest = np.where(nc2test_array==cBest)[0][0]
            except:
                IdxBest = 0
    
        return classi[IdxBest][1], classi[IdxBest][0], Centers[IdxBest] 
    

def Silhouette(Data, Clusters, Distance, pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate):
    """
    silhouette score of the Clusters given the distance and the parameters of the distance.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        Clusters (tuple): number of Clusters and Clusters obtained by the clustering Algorithm applied.
        Distance (str): metric used to compute distances.
        pMinkowski (int): Defaults to 2.
        wMax (int): upper bound of the weights. Defaults to 1.
        g (int): exponential parameter. Defaults to 1.
        epsilonEDR (float): Defaults to 0.001.
        epsilonLCSS (float): Defaults to 0.001.
        SamplingRate (float): Defaults to 1000.
    """
    labels = np.array(range(Data.shape[0]))
    for j in range(Clusters[0]):
        labels[Clusters[1][j]] = j+1
    if len(np.unique(labels))>1:
        d = 'd_'+Distance
        d = globals()[d]
        d.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
        score = silhouette_score(Data, labels, metric=d)
    else:
        score = 0 

    return score

def ICA_Algo(Data, ncomp = 10):
    """
    Apply Independent Component Analysis to reduce dimensionality and reconstruct the Data.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset (n_samples x n_features).
        ncomp (int): maximum number of independent components to compute. Defaults to 10.
    
    Returns:
        Xtransformed (np.ndarray): Dataset projected on the selected independent components.
        Xback (np.ndarray): reconstructed Dataset projected back to the original feature space.
    """
    n = min(Data.shape[0], Data.shape[1], ncomp)
    ICA = FastICA(n_components=n, random_state=0)
    Xtransformed = ICA.fit_transform(Data)
    Xback = ICA.inverse_transform(Xtransformed)
    return Xtransformed, Xback 

def kernelPCA_Algo(Data, ncomp = 10):
    """
    Given a Dataset and a threshold between (0 and 1) in the term of the dispersion of the Data to maintain, the function applies.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        ncomp (int): componentes number. Defaults to 10.
    
    Returns:
        Xtransformed (np.ndarray): the Dataset projected on the principal components selected from the kernel PCA.
        Xback (np.ndarray): the Dataset projected back on the original Dataset.
    """
    scaler = StandardScaler()
    DataScaled = scaler.fit_transform(Data)
    n = min(Data.shape[0], Data.shape[1], ncomp)
    pca = KernelPCA(n_components=n, fit_inverse_transform=True)
    pca.fit(DataScaled)
    Xtransformed = pca.fit_transform(DataScaled)
    Xback = pca.inverse_transform(Xtransformed)
    Xback = scaler.inverse_transform(Xback)

    return Xtransformed, Xback

def PCA_Algo(Data, ThresholdVariance = 0.9):
    """
    Given a Dataset and a threshold between (0 and 1) in the term of the dispersion of the Data to maintain, the function applies.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        ThresholdVariance (float): A number between 0 and 1 that represent the amount of the dispersion of Data to maintain apllying the PCA. Defaults to 0.9.
    
    Returns:
        Xtransformed (np.ndarray): the Dataset projected on the principal components selected from the PCA.
        Xback (np.ndarray): the Dataset projected back on the original Dataset.
    """
    n = min(Data.shape[0], Data.shape[1])
    pca = PCA(n_components=n)
    pca.fit(Data)
    variance = pca.explained_variance_ratio_
    sum_ratio = 0
    i = 0
    while sum_ratio < ThresholdVariance:
        sum_ratio += variance[i]
        i += 1
    n_c = i
    #print("Number of components to select: " +str(n_c))
    pca = PCA(n_components=n_c)
    Xtransformed  = pca.fit_transform(Data)
    Xback = pca.inverse_transform(Xtransformed)

    return Xtransformed, Xback

def Leiden_Algo(Data, ThresholdLeiden=0.95, DistanceStr = 'm', pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Leiden graph based Algorithm.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        ThresholdLeiden (float): if graph based on Pearson's correlation coefficient we put to zero the weights of the edges under the threshold. Defaults to 0.95.
        DistanceStr (str): metric to compute the adjacency matrix. Defaults to 'm'.
        pMinkowski (int): Minkowski parameter. Defaults to 2.
        wMax (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilonEDR (float): EDR threshold. Defaults to 0.001.
        epsilonLCSS (float): LCSS threshold. Defaults to 0.001.
        SamplingRate (int): STS parameter. Defaults to 1000.
    
    Returns:
        Clusters (list): Clusters of the applied Algorithm.
        G (ig graph): Leiden graph.
        partition (ig graph object): object containing the Clusters labels.
    """
    #distance ='rho'
    if DistanceStr =='rho':
        df = pd.DataFrame(Data)
        c = df.corr()
        c[c<=ThresholdLeiden]=0

    else:
        c = AdjacencyMatrix(Data.T, DistanceStr, pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate) 

    G =ig.Graph.Weighted_Adjacency(c, mode='undirected', attr='weight', loops=False)
    partition=la.find_partition(G, la.ModularityVertexPartition)
    optimiser = la.Optimiser()
    improvement = optimiser.optimise_partition(partition)
    while improvement:
        improvement = optimiser.optimise_partition(partition)
    partition_membership=partition.membership
    nClusters = max(partition_membership)+1
    partition_membership = np.array(partition_membership)
    Clusters =[[] for i in range(nClusters)] 
    for i in range(nClusters):
        idx = np.where(partition_membership==i)[0]
        Clusters[i].append(idx) 

    return Clusters, G, partition

def Clustering(Data, Algo = 'KM', DistanceStr = 'm', methodHC = 'complete', criterionHC = 'distance', methodKM = 'silhouette', MaxIterFCM=10, ThresholdVariance = 0.9, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, FuzzyParameter = 1, ThresholdDendrogram = 0.7, MaxClasses = [2], ThresholdLeiden = 0.9, SamplingRate = 1000, pMinkowski  = 2, Normalization = 'OFF', NormMode ='min_max_single', ica_ncomp=10, kpca_ncomp=10): 
    """
    Given user choice, clustering Algorithm.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        Algo (str): clustering Algorithm. Choices: c-means ('KM'), fuzzy c-means ('FCM'), hierarchical clustering ('HC'), Leiden ('Leiden') and applying first a dimensionality reduction by PCA ('PCA&KM', 'PCA&FCM', 'PCA&HC', 'PCA&Leiden'). Defaults to 'KM'.
        DistanceStr (str): metric used for clustering. Defaults to 'm'.
        methodHC (str): linkage method. Choices: 'complete', 'single', 'average'. Defaults to 'complete'.
        criterionHC (str): Hierarchical clustering criterion. Choices: 'distance', 'maxclust'. Defaults to 'distance'.
        methodKM (str): Method to selct the optimal number of centroid. 'Choices: 'silhouette', 'wcss'. Defaults to 'silhouette'.
        MaxIterFCM (int): maximum number of iterations for FCM. Defaults to 10.
        ThresholdVariance (float): explained variance after PCA. Defaults to 0.9.
        wMax (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilonEDR (float): EDR threshold. Defaults to 0.001.
        epsilonLCSS (float): LCSS threshold. Defaults to 0.001.
        FuzzyParameter (int): FCM parameter. Defaults to 1.
        ThresholdDendrogram (float): Cut height in percentage. Defaults to 0.7.
        MaxClasses (list): Classes to test. Defaults to [2].
        ThresholdLeiden (float): Leiden threshold. Defaults to 0.9.
        SamplingRate (int): STS parameter. Defaults to 1000.
        pMinkowski (int): Minkowski parameter. Defaults to 2.
        Normalization (str): To applying Normalization. Choices: 'ON', 'OFF'. Defaults to 'OFF'.
        NormMode (str): If Normalization applied, to select the modality. Choices: 'min_max_single', 'min_max_global', 'mu_std_single', 'mu_std_global'. Defaults to 'min_max_single'.
        ica_ncomp (int): the number of components for ICA.
        kpca_ncomp (int): the number of components for kernelPCA.
    """

    distance = 'd_'+DistanceStr
    distance = globals()[distance]
    distance.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)

   
    n = Data.shape[0]
    x = range(Data.shape[1])
        
    Data_plot = Data.copy()

    # Normalization
    if Normalization == 'ON':
        if NormMode =='min_max_single':
            Data = Whitening(Data)
        elif NormMode =='min_max_global':
            Data = WhiteningGlobal(Data)
        elif NormMode =='mu_std_single':
            Data = Whitening(Data)
        else:
            Data = WhiteningGlobal(Data)


    if type(MaxClasses) == int:
        if MaxClasses < 2:
            MaxClasses = 2
        if MaxClasses >= len(Data):
            MaxClasses = len(Data)
        nc2test = [MaxClasses]
    else:
        if len(MaxClasses)==1:
            MaxClasses = MaxClasses[0] 
            if MaxClasses < 2:
                MaxClasses = 2
            if MaxClasses >= len(Data):
                MaxClasses = len(Data)
            nc2test = [MaxClasses]
        else: 
            nc2test = MaxClasses 

    if n==1:
        return (1, [[0]])
    else: 
        if Algo == "HC":
            ThresholdDendrogram = Dendrogram(Data=Data, methodHC=methodHC, distance=distance, ThresholdDendrogram=ThresholdDendrogram)
            ClustersHC = HierarchicalClustering(Data=Data, methodHC=methodHC, distance=distance, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=nc2test, criterion = criterionHC,
                                                 DistanceStr = DistanceStr,
                                                 pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate = SamplingRate)
            NClasses = len(ClustersHC)
            Clusters = []
            for i in range(NClasses):
                if len(ClustersHC[i])>0:
                    Clusters.append(ClustersHC[i])
            NClasses = len(Clusters)
        
        elif Algo == "KM":  
            Clusters, NClasses, Centers = Kmeans_Algo(Data=Data, nc2test=nc2test, distance=distance, methodKM=methodKM)

        
        elif Algo == "FCM":
            ClustersKM, NClassesKM, CentersKM = Kmeans_Algo(Data=Data, nc2test=nc2test, distance=distance, methodKM=methodKM)
            Clusters, Centers, MembershipMat = FCM.FCM(Data=Data, NClasses = NClassesKM, Centers=CentersKM, FuzzyParameter=FuzzyParameter, MaxIter=MaxIterFCM, Metric=distance)
            NClasses = len(Clusters)

        elif Algo == "KShape":
            Clusters, NClasses, Centers = Kshape_Algo(Data=Data, nc2test=nc2test, methodKM=methodKM)

        elif Algo == "PCA&KShape":
            Data, DataPostPCA = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            Clusters, NClasses, Centers = Kshape_Algo(Data=Data, nc2test=nc2test, methodKM=methodKM)

        elif Algo == "ICA&KShape":
            Data, DataPostICA = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            Clusters, NClasses, Centers = Kshape_Algo(Data=Data, nc2test=nc2test, methodKM=methodKM)

        elif Algo == "KernelPCA&KShape":
            Data, DataPostPCA = kernelPCA_Algo(Data=Data, ncomp = kpca_ncomp)
            Clusters, NClasses, Centers = Kshape_Algo(Data=Data, nc2test=nc2test, methodKM=methodKM)

    
    
        elif Algo=="Leiden":
            Clusters_L = Leiden_Algo(Data=Data.T, ThresholdLeiden=ThresholdLeiden, DistanceStr=DistanceStr, pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate=SamplingRate)[0]
            NClasses = len(Clusters_L)
            Clusters = []
            for i in range(NClasses):
                if len(Clusters_L[i][0])>0:
                    Clusters.append(Clusters_L[i][0])
            NClasses = len(Clusters)


        elif Algo=="PCA&HC":
            Data, DataPostPCA  = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            ThresholdDendrogram = Dendrogram(Data=Data, methodHC=methodHC, distance=distance, ThresholdDendrogram=ThresholdDendrogram)
            ClustersHC = HierarchicalClustering(Data=Data, methodHC=methodHC, distance=distance, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=nc2test, criterion=criterionHC,
                                                 DistanceStr=DistanceStr,
                                                 pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate = SamplingRate)
            NClasses = len(ClustersHC)
            
            Clusters = []
            for i in range(NClasses):
                if len(ClustersHC[i])>0:
                    Clusters.append(ClustersHC[i])
            NClasses = len(Clusters)

        elif Algo == "PCA&KM":
            Data, DataPostPCA = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            Clusters, NClasses, Centers = Kmeans_Algo(Data=Data, nc2test=nc2test, distance=distance, methodKM=methodKM)
          
            
        
        elif Algo == "PCA&FCM":
            Data, DataPostPCA  = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            ClustersKM, NClassesKM, CentersKM = Kmeans_Algo(Data=Data, nc2test=nc2test, distance=distance, methodKM=methodKM)
            MaxIter = 5
            Clusters, Centers_FCM, membership_mat = FCM.FCM(Data=Data, NClasses = NClassesKM, Centers=CentersKM, FuzzyParameter=FuzzyParameter, MaxIter=MaxIterFCM, Metric=distance)
            NClasses = len(Clusters)


        elif Algo =="PCA&Leiden":
            Data, DataPostPCA  = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            Clusters_L = Leiden_Algo(Data=Data.T, ThresholdLeiden=ThresholdLeiden, DistanceStr=DistanceStr, pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate=SamplingRate)[0]
            NClasses = len(Clusters_L)
            Clusters = []
            for i in range(NClasses):
                if len(Clusters_L[i][0])>0:
                    Clusters.append(Clusters_L[i][0])
            NClasses = len(Clusters)

        elif Algo=="ICA&HC":
            Data, DataPostICA = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            ThresholdDendrogram = Dendrogram(Data=Data, methodHC=methodHC, distance=distance, ThresholdDendrogram=ThresholdDendrogram)
            ClustersHC = HierarchicalClustering(Data=Data, methodHC=methodHC, distance=distance, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=nc2test, criterion=criterionHC,
                                                 DistanceStr = DistanceStr,
                                                 pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate = SamplingRate)
            NClasses = len(ClustersHC)
            Clusters = []
            for i in range(NClasses):
                if len(ClustersHC[i])>0:
                    Clusters.append(ClustersHC[i])
            NClasses = len(Clusters)

        
        elif Algo == "ICA&KM":
            Data, DataPostICA  = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            Clusters, NClasses, Centers = Kmeans_Algo(Data=Data, nc2test=nc2test, distance=distance, methodKM=methodKM)

            
        elif Algo == "ICA&FCM":
            Data, DataPostICA   = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            ClustersKM, NClassesKM, CentersKM = Kmeans_Algo(Data=Data, nc2test=nc2test, distance=distance, methodKM=methodKM)
            MaxIter = 5
            Clusters, CentersFCM, MembershipMat = FCM.FCM(Data=Data, NClasses = NClassesKM, Centers=CentersKM, FuzzyParameter=FuzzyParameter, MaxIter=MaxIterFCM, Metric=distance)
            NClasses = len(Clusters)


        elif Algo =="ICA&Leiden":
            Data, DataPostICA   = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            Clusters_L = Leiden_Algo(Data=Data.T, ThresholdLeiden=ThresholdLeiden, DistanceStr=DistanceStr, pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate=SamplingRate)[0]
            NClasses = len(Clusters_L)
            Clusters = []
            for i in range(NClasses):
                if len(Clusters_L[i][0])>0:
                    Clusters.append(Clusters_L[i][0])
            NClasses = len(Clusters)
            
        elif Algo=="kernelPCA&HC":
            Data, DataPostPCA  = kernelPCA_Algo(Data=Data, ncomp = kpca_ncomp)
            ThresholdDendrogram = Dendrogram(Data=Data, methodHC=methodHC, distance=distance, ThresholdDendrogram=ThresholdDendrogram)
            ClustersHC = HierarchicalClustering(Data=Data, methodHC=methodHC, distance=distance, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=nc2test, criterion=criterionHC,
                                                 DistanceStr = DistanceStr,
                                                 pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate = SamplingRate)
            NClasses = len(ClustersHC)
            Clusters = []
            for i in range(NClasses):
                if len(ClustersHC[i])>0:
                    Clusters.append(ClustersHC[i])
            NClasses = len(Clusters)

        
        elif Algo == "kernelPCA&KM":
            Data, DataPostPCA = kernelPCA_Algo(Data=Data, ncomp = kpca_ncomp)
            Clusters, NClasses, Centers = Kmeans_Algo(Data=Data, nc2test=nc2test, distance=distance, methodKM=methodKM)
          
            
        
        elif Algo == "kernelPCA&FCM":
            Data, DataPostPCA  = kernelPCA_Algo(Data=Data, ncomp = kpca_ncomp)
            ClustersKM, NClassesKM, CentersKM = Kmeans_Algo(Data=Data, nc2test=nc2test, distance=distance, methodKM=methodKM)
            MaxIter = 5
            Clusters, CentersFCM, MembershipMat = FCM.FCM(Data=Data, NClasses = NClassesKM, Centers=CentersKM, FuzzyParameter=FuzzyParameter, MaxIter=MaxIterFCM, Metric=distance)
            NClasses = len(Clusters)

        return (NClasses, Clusters)

def RecursiveClustering(Data, Algo = 'KM', DistanceStr = 'm', methodHC = 'complete', criterionHC = 'distance', methodKM = 'silhouette', MaxIterFCM=10, ThresholdVariance = 0.9, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, FuzzyParameter = 1, noise = 0, ThresholdDendrogram = 0.33, MaxClasses = [2], ThresholdLeiden = 0.9, SamplingRate = 1000, pMinkowski  = 2, Normalization = 'OFF', NormMode ='min_max_single'):
    """
    This Algorithm is recursive, based on the sum of squares criteria.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        Algo (str): clustering Algorithm. Choices: c-means ('KM'), fuzzy c-means ('FCM'), hierarchical clustering ('HC'), Leiden ('Leiden') and applying first a dimensionality reduction by PCA ('PCA&KM', 'PCA&FCM', 'PCA&HC', 'PCA&Leiden'). Defaults to 'KM'.
        DistanceStr (str): metric used for clustering. Defaults to 'm'.
        methodHC (str): linkage method. Choices: 'complete', 'single', 'average'. Defaults to 'complete'.
        criterionHC (str): Hierarchical clustering criterion. Choices: 'distance', 'maxclust'. Defaults to 'distance'.
        methodKM (str): Method to selct the optimal number of centroid. 'Choices: 'silhouette', 'wcss'. Defaults to 'silhouette'.
        MaxIterFCM (int): maximum number of iterations for FCM. Defaults to 10.
        ThresholdVariance (float): explained variance after PCA. Defaults to 0.9.
        wMax (int): WDTW and WDDTW parameter. Defaults to 1.
        g (int): WDTW and WDDTW parameter. Defaults to 1.
        epsilonEDR (float): EDR threshold. Defaults to 0.001.
        epsilonLCSS (float): LCSS threshold. Defaults to 0.001.
        FuzzyParameter (int): FCM parameter. Defaults to 1.
        noise (int): Percentage of noise to add to Data. Defaults to 0.
        ThresholdDendrogram (float): Cut height in percentage. Defaults to 0.7.
        MaxClasses (int): maximum number of classes. Defaults to 1.
        ThresholdLeiden (float): Leiden threshold. Defaults to 0.9.
        SamplingRate (int): STS parameter. Defaults to 1000.
        pMinkowski (int): Minkowski parameter. Defaults to 2.
        Normalization (str): To applying Normalization. Choices: 'ON', 'OFF'. Defaults to 'OFF'.
        NormMode (str): If Normalization applied, to select the modality. Choices: 'min_max_single', 'min_max_global', 'mu_std_single', 'mu_std_global'. Defaults to 'min_max_single'.
    """
    
    distance = 'd_'+DistanceStr
    distance = globals()[d]
        
    Data_plot = Data.copy()

    distance.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
    media = np.mean(Data_plot,0)
    SS = 0
    for i in range(Data_plot.shape[0]):
        SS += d(Data_plot[i], media)**2
    Clusters = Clustering(Data=Data, Algo=Algo, DistanceStr=DistanceStr, methodHC = methodHC, criterionHC=criterionHC, methodKM=methodKM, MaxIterFCM=MaxIterFCM, ThresholdVariance=ThresholdVariance, wMax=wMax, g=g, epsilonEDR=epsilonEDR, epsilonLCSS=epsilonLCSS, FuzzyParameter=FuzzyParameter, noise=noise, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=MaxClasses, ThresholdLeiden=ThresholdLeiden, SamplingRate=SamplingRate, pMinkowski =pMinkowski , Normalization=Normalization, NormMode=NormMode) 
    Clusters = Clusters[1]    
    wcssk_list = [] 
    wcss_value = 0
    for j in range(len(Clusters)):
        wcssk = 0
        cluster_points = Data_plot[Clusters[j]] 
        cluster_points=cluster_points.reshape((cluster_points.shape[-2],cluster_points.shape[-1]))
        center = np.mean(cluster_points,0)
        for i in range(cluster_points.shape[0]): 
            wcssk += d(cluster_points[i], center)**2
        wcss_value+=wcssk
        wcssk_list.append(wcssk)
    while max(wcssk_list)>SS/100*25:
        new_Clusters = [] 
        for j in range(len(Clusters)):
            if wcssk_list[j]>SS/100*25:
                Data = Data_plot.copy()
                Clusters_j = Clustering(Data=Data[Clusters[j]].reshape(Data[Clusters[j]].shape[-2],Data[Clusters[j]].shape[-1]), Algo=Algo, DistanceStr=DistanceStr, methodHC = methodHC, criterionHC=criterionHC, methodKM=methodKM, MaxIterFCM=MaxIterFCM, ThresholdVariance=ThresholdVariance, wMax=wMax, g=g, epsilonEDR=epsilonEDR, epsilonLCSS=epsilonLCSS, FuzzyParameter=FuzzyParameter, noise=noise, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=MaxClasses, ThresholdLeiden=ThresholdLeiden, SamplingRate=SamplingRate, pMinkowski =pMinkowski , Normalization=Normalization, NormMode=NormMode)
                Clusters_j = Clusters_j[1] 
                indexes = np.array(sorted(Clusters[j]))
                for i in range(len(Clusters_j)):
                    new_Clusters.append(list(indexes[Clusters_j[i]]))
            else:
                new_Clusters.append(Clusters[j])
        Clusters = new_Clusters
        wcssk_list = [] 
        wcss_value = 0
        for j in range(len(Clusters)):
            wcssk = 0
            cluster_points = Data_plot[Clusters[j]] 
            cluster_points=cluster_points.reshape((cluster_points.shape[-2],cluster_points.shape[-1]))
            center = np.mean(cluster_points,0)
            for i in range(cluster_points.shape[0]): 
                wcssk += distance(cluster_points[i], center)**2
            wcss_value+=wcssk
            wcssk_list.append(wcssk)
    
    return (len(Clusters),Clusters)
    
def ClusterCentroids(Data, Clusters):
    """
    Given a Dataset and its subdivision in Clusters, the function returns the Centers of the Clusters.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        Clusters (tuple): number of Clusters and Clusters obtained by the clustering Algorithm applied.
    
    Returns:
        Centroids (np.ndarray): a vector with the coordinates of the Centers of the Clusters.
    """
    NClasses = len(Clusters[1])
    Centroids = np.zeros((NClasses, Data.shape[1]))
    for i in range(NClasses):
        Centroids[i]=np.mean(Data[Clusters[1][i]],0)
    return Centroids 

def Classification(Centroids, Data, DistanceStr='m', pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Given a Dataset, a group of Centroids and a metric to compute distances between each Data and each Centroids,.
    
    Args:
        Centroids (np.ndarray): a vector with the coordinates of the Centers of some Clusters.
        Data (np.ndarray): 2D matrix representing the Dataset.
        DistanceStr (str): metric to compute distances. Defaults to 'ed'.
    
    Returns:
        classification (list): Clusters obtained from the Dataset assigning each Data to the class of the closest centroid.
    """
    distance = 'd_'+DistanceStr
    metric = globals()[distance]
    metric.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
    n = Data.shape[0]
    x = Data.shape[1] 
    c = Centroids.shape[0] 
    classification = []  
    for i in range(n):
        distances = [] 
        for j in range(c):
            distances.append(metric(Data[i],Centroids[j]))
        distances = np.array(distances)
        idx_m = np.where(distances==min(distances))[0][0] 
        idx_m = int(idx_m)
        classification.append(idx_m)

    return classification

def GaussianNoise(Data, noise, seed):
    """
    Function to add Gaussian noise to Data.
    
    Args:
        Data (np.ndarray): 2D matrix representing the Dataset.
        noise (float): amount of noise to add.
        seed (int): seed for random generation.
    
    Returns:
        Data (np.ndarray): 2D matrix representing the Dataset after the adding of noise.
    """
    np.random.seed(seed)
    for i in range(len(Data)):
        sigma = np.std(Data[i])
        if sigma == 0:
            Data[i] = Data[i]+noise*np.random.normal(0,1,len(Data[i]))
        else:
            Data[i] = Data[i]+noise*np.random.normal(0,sigma,len(Data[i]))

    return Data
