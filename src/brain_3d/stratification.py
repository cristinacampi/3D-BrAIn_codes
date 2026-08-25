"Stratification functions"
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
from . import Fcm
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

    Parameters
    ----------
    a : np.ndarray
        first time series.
    b : np.ndarray
        second time series.
    wMax : int
        Unused. Defaults to 1.
    pMinkowski : int
        parameter. Defaults to 2 (euclidean distance).
    g : int
        Unused. Defaults to 1.
    epsilonEDR : float
        Unused. Defaults to 0.001.
    epsilonLCSS : float
        Unused. Defaults to 0.001.
    SamplingRate : float
        Unused. Defaults to 1000.

    Returns
    -------
    d : float
        Minkowski distance between a and b.
    """
    L1 = len(a)
    L2 = len(b)
    if L1 != L2:
        print('vectors with not equal length')
    else:
        D = minkowski(a, b, p=pMinkowski )

        return D

def MatrixC(M):
    """
    Creating the matrix C helpful in the dtw (and derivatives) distance.

    Parameters
    ----------
    M : np.ndarray
        matrix of the distances between the entries of two vectors.

    Returns
    -------
    C : np.ndarray
        matrix helpful in the dtw (and derivatives) distance.
    """
    L1 = M.shape[0]
    C = np.zeros((L1+1, L1+1))
    C[:][:] = np.inf
    C[0][0] = 0
    for I in range(L1):
        IC = I+1
        for J in range(L1):
            JC = J+1
            C[IC][JC] = M[I][J] + min(C[IC-1][JC-1], C[IC-1][JC], C[IC][JC-1])
    return C

def Warping(a, b, M):
    """
    Realignment of a signal a on a signal b (or vice versa).

    Parameters
    ----------
    a : np.ndarray
        array 1D.
    b : np.ndarray
        array 1D.
    M : np.ndarray
        2D matrix containing the punctual ed distances between the entries of the vectors a and b.

    Returns
    -------
    tuple
        realignments and indexes of the realignment.
    """
    C = MatrixC(M)
    I = C.shape[0]-1
    J = C.shape[1]-1
    L = []
    while (I>0) & (J>0):
        L.append((I,J))
        MinCost = min(C[I-1][J],C[I][J-1], C[I-1][J-1])
        if MinCost == C[I-1][J-1]:
            I = I-1
            J = J-1
        elif MinCost == C[I][J-1]:
            I = I
            J = J-1
        elif MinCost == C[I-1][J]:
            I = I-1
            J = J
    IdxA = []
    IdxB = []
    for K in range(len(L)):
        IdxA.append(L[len(L)-1-K][0])
        IdxB.append(L[len(L)-1-K][1])

    X = range(len(a))
    WA = np.zeros(len(IdxA))
    WB = np.zeros(len(IdxA))
    for I in range(len(IdxA)):
        WA[I] = a[IdxA[I]-1]
        WB[I] = b[IdxB[I]-1]
    IdxA = np.array(IdxA)-1
    IdxB = np.array(IdxB)-1


    return WA, WB, IdxA, IdxB

def MatrixM(a, b):
    """
    2D matrix containing the punctual ed distances between the entries of the vectors a and b.

    Parameters
    ----------
    a : np.ndarray
        vector of length n.
    b : np.ndarray
        vector of length n.

    Returns
    -------
    M : np.ndarray
        2D matrix containing the punctual ed distances between the entries of the vectors a and b.
    """
    L1 = len(a)
    '''
    M = np.zeros((l1, l1))
    for i in range(l1):
        for j in range(l1):
            M[i][j] = (a[i]-b[j])**2
    '''
    Aa = np.repeat(a,L1,axis=0)
    Aa = np.reshape(Aa,(L1,L1))
    Bb = np.repeat(b,L1,axis=0)
    Bb = np.reshape(Bb,(L1,L1))
    Bb = Bb.T
    M = (Aa-Bb)**2


    return M

def d_dtw(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Dynamic Time Warping distance.

    Parameters
    ----------
    a : np.ndarray
        first time series.
    b : np.ndarray
        second time series.
    pMinkowski : int
        Unused. Defaults to 2.
    wMax : int
        Unused. Defaults to 1.
    g : int
        Unused. Defaults to 1.
    epsilonEDR : float
        Unused. Defaults to 0.001.
    epsilonLCSS : float
        Unused. Defaults to 0.001.
    SamplingRate : float
        Unused. Defaults to 1000.

    Returns
    -------
    d : float
        dtw distance between a and b.
    """
    L1 = len(a)
    L2 = len(b)
    if L1 != L2:
        print('vectors with not equal length')
    else:
        M = MatrixM(a, b)
        C = MatrixC(M)
        D = C[L1][L1]
        return math.pow(D,0.5)

def a1_b1_ddtw(a, b):
    """
    The new vectors built for the ddtw (wddtw) distance of the original vectors.

    Parameters
    ----------
    a : np.ndarray
        vector of length n.
    b : np.ndarray
        vector of length n.

    Returns
    -------
    tuple
        the new vectors for the ddtw (wddtw) distance between a and b.
    """
    L1 = len(a)
    A1 = np.empty(L1-2)
    B1 = np.empty(L1-2)
    Indexes =np.array(range(L1-2))+1
    for I in Indexes:
        A1[I-1]=((a[I]-a[I-1])+((a[I+1]-a[I-1])/2))/2
        B1[I-1]=((b[I]-b[I-1])+((b[I+1]-b[I-1])/2))/2
    return A1, B1

def d_ddtw(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Derivative Dynamic Time Warping distance.

    Parameters
    ----------
    a : np.ndarray
        first time series.
    b : np.ndarray
        second time series.
    pMinkowski : int
        Unused. Defaults to 2.
    wMax : int
        Unused. Defaults to 1.
    g : int
        Unused. Defaults to 1.
    epsilonEDR : float
        Unused. Defaults to 0.001.
    epsilonLCSS : float
        Unused. Defaults to 0.001.
    SamplingRate : float
        Unused. Defaults to 1000.

    Returns
    -------
    d : float
        ddtw distance between a and b.
    """
    L1 = len(a)
    L2 = len(b)
    if L1 != L2:
        print('vectors with not equal length')
    else:
        A1, B1 = a1_b1_ddtw(a, b)
        D = d_dtw(A1,B1)
        return D

def d_wdtw(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Weighted Dynamic Time Warping distance.

    Parameters
    ----------
    a : np.ndarray
        first time series.
    b : np.ndarray
        second time series.
    pMinkowski : int
        Unused. Defaults to 2.
    wMax : int
        upper bound of the weights. Defaults to 1.
    g : int
        exponential parameter. Defaults to 1.
    epsilonEDR : float
        Unused. Defaults to 0.001.
    epsilonLCSS : float
        Unused. Defaults to 0.001.
    SamplingRate : float
        Unused. Defaults to 1000.

    Returns
    -------
    d : float
        wdtw distance between a and b.
    """
    L1 = len(a)
    L2 = len(b)
    if L1 != L2:
        print('vectors with not equal length')
    else:
        M = MatrixM(a, b)
        M = MatrixMw(M, wMax, g)
        C = MatrixC(M)
        D = C[L1][L1]
        return math.pow(D,0.5)

def MatrixMw(M, wMax, g):
    """
    Matrix of the punctual distances between the entries of two vectors but with weights based on the indexes of the entries.

    Parameters
    ----------
    M : np.ndarray
        matrix of the ed distances between the entries of two vectors.
    wMax : float
        parameter for the weights.
    g : float
        parameter for the weights.

    Returns
    -------
    Mw : np.ndarray
        Matrix of the punctual distances between the entries of two vectors but with weights based on the indexes of the entries.
    """
    L1=M.shape[0]

    for I in range(L1):
            for J in range(L1):
                M[I,J] = (wMax/(1+math.exp(-g*(abs(I-J)-L1/2))))*M[I,J]
    return M

def d_wddtw(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate = 1000):
    """
    Derivative Weigthed Dynamic Time Warping distance.

    Parameters
    ----------
    a : np.ndarray
        first time series.
    b : np.ndarray
        second time series.
    pMinkowski : int
        Unused. Defaults to 2.
    wMax : int
        upper bound of the weights. Defaults to 1.
    g : int
        exponential parameter. Defaults to 1.
    epsilonEDR : float
        Unused. Defaults to 0.001.
    epsilonLCSS : float
        Unused. Defaults to 0.001.
    SamplingRate : float
        Unused. Defaults to 1000.

    Returns
    -------
    d : float
        wdtw distance between a and b.
    """
    A1, B1 = a1_b1_ddtw(a, b)
    D = d_wdtw(A1, B1, wMax = wMax, g = g)
    return D

def d_lcss(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Longest Common Subsequence distance.

    Parameters
    ----------
    a : np.ndarray
        first time series.
    b : np.ndarray
        second time series.
    pMinkowski : int
        Unused. Defaults to 2.
    wMax : int
        Unused. Defaults to 1.
    g : int
        Unused. Defaults to 1.
    epsilonEDR : float
        Unused. Defaults to 0.001.
    epsilonLCSS : float
        Threshold. Defaults to 0.001.
    SamplingRate : float
        Unused. Defaults to 1000.

    Returns
    -------
    d : float
        lcss distance between a and b.
    """
    EpsilonLCSSAbs = epsilonLCSS*np.linalg.norm(a)

    L1 = len(a)
    L2 = len(b)
    if L1 != L2:
        print('vectors with not equal length')
    else:
        L = np.zeros((L1+1, L1+1))
        for I in range(L1):
            IL = I+1
            for J in range(L1):
                JL = J+1
                if abs(a[I]-b[J])<EpsilonLCSSAbs:
                    L[IL][JL] = L[IL-1][JL-1]+1
                else:
                    L[IL][JL] = max(L[IL-1][JL], L[IL][JL-1])
        LCSS = L[L1][L1]
        D = 1 - LCSS/L1
        return D

def d_edr(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Edit distance on Real Sequences.

    Parameters
    ----------
    a : np.ndarray
        first time series.
    b : np.ndarray
        second time series.
    pMinkowski : int
        Unused. Defaults to 2.
    wMax : int
        Unused. Defaults to 1.
    g : int
        Unused. Defaults to 1.
    epsilonEDR : float
        Threshold. Defaults to 0.001.
    epsilonLCSS : float
        Unused. Defaults to 0.001.
    SamplingRate : float
        Unused. Defaults to 1000.

    Returns
    -------
    d : float
        edr distance between a and b.
    """
    EpsilonEDRAbs = epsilonEDR*np.linalg.norm(a)

    L1 = len(a)
    L2 = len(b)
    if L1 != L2:
        print('vectors with not equal length')
    else:
        E = np.zeros((L1+1, L1+1))

        for I in range(L1):
            IE = I+1
            for J in range(L1):
                JE = J+1
                if abs(a[I]-b[J])<EpsilonEDRAbs:
                    C = 0
                else:
                    C = 1
                Match = E[IE-1][JE-1]+C
                Insert = E[IE-1][JE]+1
                Delete = E[IE][JE-1]+1
                E[IE][JE] = min(Match, Insert, Delete)
        D = E[L1][L1]
        return D

def d_rho_2 (a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Distance based on the Pearson's correlation coefficient.

    Parameters
    ----------
    a : np.ndarray
        first time series.
    b : np.ndarray
        second time series.
    pMinkowski : int
        Unused. Defaults to 2.
    wMax : int
        Unused. Defaults to 1.
    g : int
        Unused. Defaults to 1.
    epsilonEDR : float
        Unused. Defaults to 0.001.
    epsilonLCSS : float
        Unused. Defaults to 0.001.
    SamplingRate : float
        Unused. Defaults to 1000.

    Returns
    -------
    d : float
        rho2 distance between a and b.
    """

    L1 = len(a)
    L2 = len(b)
    if L1 != L2:
        print('vectors with not equal length')
    else:
        Rho = pearsonr(a,b)[0]
        D = 2*(1-Rho)
    return D

def d_sts(a, b, pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Short Time Series distance.

    Parameters
    ----------
    a : np.ndarray
        first time series.
    b : np.ndarray
        second time series.
    pMinkowski : int
        Unused. Defaults to 2.
    wMax : int
        Unused. Defaults to 1.
    g : int
        Unused. Defaults to 1.
    epsilonEDR : float
        Unused. Defaults to 0.001.
    epsilonLCSS : float
        Unused. Defaults to 0.001.
    SamplingRate : float
        Sampling Frequency. Defaults to 1000.

    Returns
    -------
    d : float
        sts distance between a and b.
    """

    if len(a) != len(b):
        print('vectors with not equal length')
    else:
        Aa = np.diff(a)
        Bb = np.diff(b)
        Aux = ((Aa-Bb)*SamplingRate)**2
        D = math.sqrt(np.sum(Aux))
        return D

def AdjacencyMatrix(Data, DistanceStr = 'm', pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Adjacency matrix for Leiden Algorithm based on metric selected.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    DistanceStr : str
        metric used to compute distances. Defaults to 'm'.
    pMinkowski : int
        Minkowski parameter. Defaults to 2.
    wMax : int
        WDTW and WDDTW parameter. Defaults to 1.
    g : int
        WDTW and WDDTW parameter. Defaults to 1.
    epsilonEDR : float
        EDR threshold. Defaults to 0.001.
    epsilonLCSS : float
        LCSS threshold. Defaults to 0.001.
    SamplingRate : int
        STS parameter. Defaults to 1000.

    Returns
    -------
    adjacency : np.ndarray
        2D matrix adjacency matrix for Leiden graph-based Algorithm.
    """

    Distance = 'd_'+DistanceStr
    Distance = globals()[Distance]
    Distance.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
    Dim = Data.shape[0]
    Matrix = np.zeros((Dim,Dim))
    for I in range(Dim):
        for J in np.array(range(I+1,Dim)):
            Matrix[I][J]=Distance(Data[I], Data[J])
    Matrix = Matrix + Matrix.T
    M = np.max(Matrix)
    Matrix2 = Matrix/M
    Adjacency = 1/(1+Matrix2)
    Adjacency[Adjacency<=0.75]=0

    return Adjacency

'''Normalization'''
def NormalizationMinMaxSingle(Data):
    """
    Normalization of a Dataset following the formula Data[i] = (Data[i]-m)/(M-m) where m and M are respectively the minimum.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix (Nobs x Ntimes) representing the Dataset.

    Returns
    -------
    Data : np.ndarray
        2D matrix (Nobs x Ntimes) representing the Dataset normalized.
    """
    for I in range(len(Data)):
        Minimum = min(Data[I,:])
        M = max(Data[I,:])
        if Minimum == M :
            Data[I,:] = Data[I,:]-Data[I,:]
        else:
            Data[I,:] = (Data[I,:]-Minimum)/(M-Minimum)

    return Data

def NormalizationMinMaxGlobal(Data):
    """
    Normalization of a Dataset following the formula Data[i] = (Data[i]-m)/(M-m) where m and M are respectively the global minimum.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix (Nobs x Ntimes) representing the Dataset.

    Returns
    -------
    Data : np.ndarray
        2D matrix (Nobs x Ntimes) representing the Dataset normalized.
    """

    Size = len(Data)
    Minimums = np.zeros(Size)
    M = np.zeros(Size)
    for I in range(Size):
        Minimums[I] = min(Data[I,:])
        M[I] = max(Data[I,:])
    Minimum = min(Minimums)
    Maximum = max(M)
    for I in range(Size):
        Data[I,:] = (Data[I,:]-Minimum)/(Maximum-Minimum)

    return Data

def Whitening(Data):
    """
    Normalization of a Dataset following the formula Data[i] = (Data[i]-mu)/sigma where mu and sigma are respectively the mean.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix (Nobs x Ntimes) representing the Dataset.

    Returns
    -------
    Data : np.ndarray
        2D matrix (Nobs x Ntimes) representing the Dataset normalized.
    """
    for I in  range(len(Data)):
        Minimum = min(Data[I,:])
        M = max(Data[I,:])
        Mu = np.mean(Data[I,:])
        Sigma = np.std(Data[I,:])
        if Minimum == M:
            Data[I,:] = Data[I,:]-Data[I,:]
        else:
            Data[I,:] = (Data[I,:]-Mu)/Sigma

    return Data

def WhiteningGlobal(Data):
    """
    Normalization of a Dataset following the formula Data[i] = (Data[i]-mu)/sigma where mu and sigma are respectively.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix (Nobs x Ntimes) representing the Dataset.

    Returns
    -------
    Data : np.ndarray
        2D matrix (Nobs x Ntimes) representing the Dataset normalized.
    """
    Size = len(Data)
    Mu = np.mean(Data)
    Sigma = np.std(Data)
    for I in range(Size):
        Data[I,:] = (Data[I,:]-Mu)/Sigma

    return Data


'''AlgoRITHMS'''

def Dendrogram(Data, Distance, methodHC ='complete', ThresholdDendrogram=0.7):
    """
    Given a Dataset, a metric, a method and a threshold, the function returns the dendrogram plot of the chosen hierarchical.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    Distance : str
        metric used to compute Distances.
    methodHC : str
        linkage method.
    ThresholdDendrogram : float
        threshold between 0 and 1 that multiplied with the height of the dendrogram represent the height at which we want to cut it.

    Returns
    -------
    max_d : float
        height of the cut of the dendrogram.
    """

    try:
        LinkageData = linkage(Data, method = methodHC, metric = Distance)
        N = len(Data)
        AggregationLevels = LinkageData[:,2]
        MaxD = ThresholdDendrogram * AggregationLevels[N-2]
    except:
        MaxD = ThresholdDendrogram*len(Data)
    '''
    plt.figure()
    plt.title("Dendrogram")
    dendrogram(LinkageData)
    plt.axhline(y = max_d, c='k')
    plt.show()
    #plt.savefig("Dendrogram.png")
    # '''
    return MaxD

def HierarchicalClustering(Data, methodHC, Distance, ThresholdDendrogram, MaxClasses, criterion, DistanceStr, pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate):
    """
    Given a Dataset, a distance, a method and a threshold, the function returns the Clusters built by the chosen hierarchical.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    methodHC : str
        linkage method.
    Distance : function
        metric used to compute distances.
    ThresholdDendrogram : float
        height at which we cut the dendrogram to form Clusters.
    MaxClasses : int
        maximum number of classes to obtain from the clustering.
    criterion : str
        clustering criterion.
    DistanceStr : str
        metric used to compute distances.
    pMinkowski : int
        Defaults to 2.
    wMax : int
        upper bound of the weights. Defaults to 1.
    g : int
        exponential parameter. Defaults to 1.
    epsilonEDR : float
        Defaults to 0.001.
    epsilonLCSS : float
        Defaults to 0.001.
    SamplingRate : float
        Defaults to 1000.

    Returns
    -------
    Clusters : list
        Clusters of the applied Algorithm.
    """
    if criterion == 'distance':
        Fclust = fclusterdata(Data, ThresholdDendrogram, criterion = 'distance', metric = Distance, method = methodHC)

    elif criterion=='maxclust':
        if type(MaxClasses) == int:
            KElbow = MaxClasses
            Fclust = fclusterdata(Data, KElbow, criterion = 'maxclust', metric = Distance, method = methodHC)
        else:
            Score = -1000
            for KElbow in MaxClasses:
                print(KElbow)
                FclustAux = fclusterdata(Data, KElbow, criterion = 'maxclust', metric = Distance, method = methodHC)
                ClustersAux = []
                for I in range(max(FclustAux)):
                    Indexes = np.where(FclustAux==I+1)[0]
                    ClustersAux.append(Indexes)
                ClustersAux = [max(FclustAux), ClustersAux]
                ScoreAux  = Silhouette(Data, ClustersAux, DistanceStr, pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
                if ScoreAux>Score:
                    Score = ScoreAux
                    Fclust = FclustAux
    Clusters = []
    for I in range(max(Fclust)):
        Indexes = np.where(Fclust==I+1)[0]
        Clusters.append(Indexes)

    return Clusters


def Kshape_Algo(Data, nc2test, methodKM='silhouette'):
    """
    Given a Dataset and the possible number of Clusters to use, the function applies the k-shape Algorithm.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset (n_samples x n_timesteps).
    nc2test : np.ndarray
        a vector with the possible choices in term of the number of Clusters.
    methodKM : str
        method to compute the optimal number of Clusters. Defaults to 'silhouette'.

    Returns
    -------
    labels : np.ndarray
        cluster labels for each sample.
    cBest : int
        optimal number of Clusters.
    Centers : np.ndarray
        coordinates of the Centers of the Clusters formed.
    """

    # KShape richiede dati normalizzati e shape (n_samples, n_timesteps, 1)
    DataScaled = TimeSeriesScalerMeanVariance().fit_transform(Data)

    Iterations = []
    BestLabels = None
    BestCenters = None
    CBest = nc2test[0]

    if methodKM == 'davies_bouldin':
        BestScore = float('inf')
    else:
        BestScore = -float('inf')

    if len(nc2test) == 1:
        NClusters = nc2test[0]
        Ks = KShape(nClusters=NClusters, random_state=42)
        Labels = Ks.fit_predict(DataScaled)
        return Labels, NClusters, Ks.cluster_Centers_

    else:
        for NClusters in nc2test:
            Ks = KShape(nClusters=NClusters, random_state=42)
            Labels = Ks.fit_predict(DataScaled)

            # WCSS con distanza shape-based non ha senso, usiamo l'inertia di KShape
            Wcss = Ks.inertia_

            if len(np.unique(Labels)) > 1:
                # per silhouette usiamo i dati originali 2D (n_samples x n_timesteps)
                Data2d = DataScaled.reshape(DataScaled.shape[0], -1)
                if methodKM == 'silhouette':
                    Score = silhouette_score(Data2d, Labels, sample_size=1000, random_state=42)
                elif methodKM == 'davies_bouldin':
                    Score = davies_bouldin_score(Data2d, Labels)
                elif methodKM == 'calinski_harabasz':
                    Score = calinski_harabasz_score(Data2d, Labels)
                elif methodKM == 'wcss':
                    Score = Wcss
            else:
                Score = 0

            if methodKM == 'davies_bouldin':
                if Score < BestScore:
                    BestScore = Score
                    CBest = NClusters
                    BestLabels = Labels
                    BestCenters = Ks.cluster_Centers_
            else:
                if Score > BestScore:
                    BestScore = Score
                    CBest = NClusters
                    BestLabels = Labels
                    BestCenters = Ks.cluster_Centers_

            print(f"For n_Centroids = {NClusters}, {methodKM} score is {Score}")
            print(f"For n_Centroids = {NClusters}, davies_bouldin score is {davies_bouldin_score(Data2d, Labels)}")
            print(f"For n_Centroids = {NClusters}, calinski_harabasz score is {calinski_harabasz_score(Data2d, Labels)}")
            print(f"For n_Centroids = {NClusters}, wcss score is {Wcss}")

            Iterations.append(Score)

        return BestLabels, CBest, BestCenters

def Kmeans_Algo(Data, nc2test, distance, methodKM = 'silhouette'):
    """
    Given a Dataset, a metric and the possible number of Centroids to use, the function applies the k-means Algorithm.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    nc2test : np.ndarray
        a vector with the possible choices in term of the number of Centroids to use to apply the Algorithm.
    distance : str
        metric used to compute distances.
    methodKM : str
        method to compute the optimal number of Centroids. Defaults to 'silhouette'.
    """


    Metric = distance_metric(type_metric.USER_DEFINED, func=distance)
    Iterations = []
    Classi = []
    Centers = []
    CBest = 1
    if methodKM == 'davies_bouldin':
        BestScore = float('inf')
    else:
        BestScore = -float('inf')

    if len(nc2test) == 1:
        NClusters = nc2test[0]
        StartCenters = kmeans_plusplus_initializer(Data, NClusters).initialize();
        KmeansInstance = kmeans(Data, StartCenters, metric=Metric)
        # run cluster analysis and obtain results
        KmeansInstance.process()
        Clusters = KmeansInstance.get_Clusters()
        Classi.append((NClusters, Clusters))
        Centers.append(KmeansInstance.get_Centers())
        return Classi[0][1], NClusters, Centers[0]

    else:
        for K in range(len(nc2test)):
            #random
            NClusters = nc2test[K]
            N = len(Data)
            StartCenters = kmeans_plusplus_initializer(Data, NClusters).initialize();
            KmeansInstance = kmeans(Data, StartCenters, metric=Metric)
            # run cluster analysis and obtain results
            KmeansInstance.process()
            Clusters = KmeansInstance.get_clusters()
            NClustersPostK = len(Clusters)
            Classi.append((NClustersPostK, Clusters))
            Centers.append(KmeansInstance.get_centers())
            Labels = np.array(range(N))
            Wcss = 0
            for J in range(len(Clusters)):
                Labels[Clusters[J]] = J+1
                ClusterPoints = Data[Clusters[J]]
                for I in range(ClusterPoints.shape[0]):
                    Wcss += distance(ClusterPoints[I], Centers[K][J])**2
            Iterations.append(Wcss)

            if len(np.unique(Labels))>1:
                if methodKM == 'silhouette':
                    Score = silhouette_score(Data, Labels, metric=Metric, sample_size=1000, random_state=42)
                elif methodKM == 'davies_bouldin':
                    Score = davies_bouldin_score(Data, Labels)
                elif methodKM == 'calinski_harabasz':
                    Score = calinski_harabasz_score(Data, Labels)
                elif methodKM == 'wcss':
                    Score = Wcss
            else:
                Score = 0

            if methodKM == 'davies_bouldin':
                if Score < BestScore:
                    BestScore = Score
                    CBest = NClusters
            else:
                if Score > BestScore:
                    BestScore = Score
                    CBest = NClusters

            print(f"For n_Centroids = {NClusters}, {methodKM} score is {Score}")
            print(f"For n_Centroids = {NClusters}, davies_bouldin score is {davies_bouldin_score(Data, Labels)}")
            print(f"For n_Centroids = {NClusters}, calinski_harabasz score is {calinski_harabasz_score(Data, Labels)}")
            print(f"For n_Centroids = {NClusters}, wcss score is {Wcss}")


            Iterations.append(Score)


        if methodKM == 'wcss':
            Nc2testArray = np.array(nc2test)
            Kl = KneeLocator(Nc2testArray, Iterations, curve="convex", direction="decreasing")
            KElbow = Kl.elbow
            if KElbow is None:
                KElbow=Nc2testArray[-1]
            IdxBest = np.where(Nc2testArray==KElbow)[0][0]
        else:
            Nc2testArray = np.array(nc2test)
            try:
                IdxBest = np.where(Nc2testArray==CBest)[0][0]
            except:
                IdxBest = 0

        return Classi[IdxBest][1], Classi[IdxBest][0], Centers[IdxBest]


def Silhouette(Data, Clusters, Distance, pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate):
    """
    silhouette score of the Clusters given the distance and the parameters of the distance.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    Clusters : tuple
        number of Clusters and Clusters obtained by the clustering Algorithm applied.
    Distance : str
        metric used to compute distances.
    pMinkowski : int
        Defaults to 2.
    wMax : int
        upper bound of the weights. Defaults to 1.
    g : int
        exponential parameter. Defaults to 1.
    epsilonEDR : float
        Defaults to 0.001.
    epsilonLCSS : float
        Defaults to 0.001.
    SamplingRate : float
        Defaults to 1000.
    """
    Labels = np.array(range(Data.shape[0]))
    for J in range(Clusters[0]):
        Labels[Clusters[1][J]] = J+1
    if len(np.unique(Labels))>1:
        D = 'd_'+Distance
        D = globals()[D]
        D.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
        Score = silhouette_score(Data, Labels, metric=D)
    else:
        Score = 0

    return Score

def ICA_Algo(Data, ncomp = 10):
    """
    Apply Independent Component Analysis to reduce dimensionality and reconstruct the Data.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset (n_samples x n_features).
    ncomp : int
        maximum number of independent components to compute. Defaults to 10.

    Returns
    -------
    Xtransformed : np.ndarray
        Dataset projected on the selected independent components.
    Xback : np.ndarray
        reconstructed Dataset projected back to the original feature space.
    """
    N = min(Data.shape[0], Data.shape[1], ncomp)
    ICA = FastICA(n_components=N, random_state=0)
    Xtransformed = ICA.fit_transform(Data)
    Xback = ICA.inverse_transform(Xtransformed)
    return Xtransformed, Xback

def kernelPCA_Algo(Data, ncomp = 10):
    """
    Given a Dataset and a threshold between (0 and 1) in the term of the dispersion of the Data to maintain, the function applies.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    ncomp : int
        componentes number. Defaults to 10.

    Returns
    -------
    Xtransformed : np.ndarray
        the Dataset projected on the principal components selected from the kernel PCA.
    Xback : np.ndarray
        the Dataset projected back on the original Dataset.
    """
    Scaler = StandardScaler()
    DataScaled = Scaler.fit_transform(Data)
    N = min(Data.shape[0], Data.shape[1], ncomp)
    Pca = KernelPCA(n_components=N, fit_inverse_transform=True)
    Pca.fit(DataScaled)
    Xtransformed = Pca.fit_transform(DataScaled)
    Xback = Pca.inverse_transform(Xtransformed)
    Xback = Scaler.inverse_transform(Xback)

    return Xtransformed, Xback

def PCA_Algo(Data, ThresholdVariance = 0.9):
    """
    Given a Dataset and a threshold between (0 and 1) in the term of the dispersion of the Data to maintain, the function applies.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    ThresholdVariance : float
        A number between 0 and 1 that represent the amount of the dispersion of Data to maintain apllying the PCA. Defaults to 0.9.

    Returns
    -------
    Xtransformed : np.ndarray
        the Dataset projected on the principal components selected from the PCA.
    Xback : np.ndarray
        the Dataset projected back on the original Dataset.
    """
    N = min(Data.shape[0], Data.shape[1])
    Pca = PCA(n_components=N)
    Pca.fit(Data)
    Variance = Pca.explained_variance_ratio_
    SumRatio = 0
    I = 0
    while SumRatio < ThresholdVariance:
        SumRatio += Variance[I]
        I += 1
    NC = I
    #print("Number of components to select: " +str(n_c))
    Pca = PCA(n_components=NC)
    Xtransformed  = Pca.fit_transform(Data)
    Xback = Pca.inverse_transform(Xtransformed)

    return Xtransformed, Xback

def Leiden_Algo(Data, ThresholdLeiden=0.95, DistanceStr = 'm', pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Leiden graph based Algorithm.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    ThresholdLeiden : float
        if graph based on Pearson's correlation coefficient we put to zero the weights of the edges under the threshold. Defaults to 0.95.
    DistanceStr : str
        metric to compute the adjacency matrix. Defaults to 'm'.
    pMinkowski : int
        Minkowski parameter. Defaults to 2.
    wMax : int
        WDTW and WDDTW parameter. Defaults to 1.
    g : int
        WDTW and WDDTW parameter. Defaults to 1.
    epsilonEDR : float
        EDR threshold. Defaults to 0.001.
    epsilonLCSS : float
        LCSS threshold. Defaults to 0.001.
    SamplingRate : int
        STS parameter. Defaults to 1000.

    Returns
    -------
    Clusters : list
        Clusters of the applied Algorithm.
    G : ig graph
        Leiden graph.
    partition : ig graph object
        object containing the Clusters labels.
    """
    #distance ='rho'
    if DistanceStr =='rho':
        Df = pd.DataFrame(Data)
        C = Df.corr()
        C[C<=ThresholdLeiden]=0

    else:
        C = AdjacencyMatrix(Data.T, DistanceStr, pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)

    G =ig.Graph.Weighted_Adjacency(C, mode='undirected', attr='weight', loops=False)
    Partition=la.find_partition(G, la.ModularityVertexPartition)
    Optimiser = la.Optimiser()
    Improvement = Optimiser.optimise_partition(Partition)
    while Improvement:
        Improvement = Optimiser.optimise_partition(Partition)
    PartitionMembership=Partition.membership
    NClusters = max(PartitionMembership)+1
    PartitionMembership = np.array(PartitionMembership)
    Clusters =[[] for I in range(NClusters)]
    for I in range(NClusters):
        Idx = np.where(PartitionMembership==I)[0]
        Clusters[I].append(Idx)

    return Clusters, G, Partition

def Clustering(Data, Algo = 'KM', DistanceStr = 'm', methodHC = 'complete', criterionHC = 'distance', methodKM = 'silhouette', MaxIterFCM=10, ThresholdVariance = 0.9, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, FuzzyParameter = 1, ThresholdDendrogram = 0.7, MaxClasses = [2], ThresholdLeiden = 0.9, SamplingRate = 1000, pMinkowski  = 2, Normalization = 'OFF', NormMode ='min_max_single', ica_ncomp=10, kpca_ncomp=10):
    """
    Given user choice, clustering Algorithm.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    Algo : str
        clustering Algorithm. Choices: c-means ('KM'), fuzzy c-means ('FCM'), hierarchical clustering ('HC'), Leiden ('Leiden') and applying first a dimensionality reduction by PCA ('PCA&KM', 'PCA&FCM', 'PCA&HC', 'PCA&Leiden'). Defaults to 'KM'.
    DistanceStr : str
        metric used for clustering. Defaults to 'm'.
    methodHC : str
        linkage method. Choices: 'complete', 'single', 'average'. Defaults to 'complete'.
    criterionHC : str
        Hierarchical clustering criterion. Choices: 'distance', 'maxclust'. Defaults to 'distance'.
    methodKM : str
        Method to selct the optimal number of centroid. 'Choices: 'silhouette', 'wcss'. Defaults to 'silhouette'.
    MaxIterFCM : int
        maximum number of iterations for FCM. Defaults to 10.
    ThresholdVariance : float
        explained variance after PCA. Defaults to 0.9.
    wMax : int
        WDTW and WDDTW parameter. Defaults to 1.
    g : int
        WDTW and WDDTW parameter. Defaults to 1.
    epsilonEDR : float
        EDR threshold. Defaults to 0.001.
    epsilonLCSS : float
        LCSS threshold. Defaults to 0.001.
    FuzzyParameter : int
        FCM parameter. Defaults to 1.
    ThresholdDendrogram : float
        Cut height in percentage. Defaults to 0.7.
    MaxClasses : list
        Classes to test. Defaults to [2].
    ThresholdLeiden : float
        Leiden threshold. Defaults to 0.9.
    SamplingRate : int
        STS parameter. Defaults to 1000.
    pMinkowski : int
        Minkowski parameter. Defaults to 2.
    Normalization : str
        To applying Normalization. Choices: 'ON', 'OFF'. Defaults to 'OFF'.
    NormMode : str
        If Normalization applied, to select the modality. Choices: 'min_max_single', 'min_max_global', 'mu_std_single', 'mu_std_global'. Defaults to 'min_max_single'.
    ica_ncomp : int
        the number of components for ICA.
    kpca_ncomp : int
        the number of components for kernelPCA.
    """

    Distance = 'd_'+DistanceStr
    Distance = globals()[Distance]
    Distance.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)


    N = Data.shape[0]
    X = range(Data.shape[1])

    DataPlot = Data.copy()

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
        Nc2test = [MaxClasses]
    else:
        if len(MaxClasses)==1:
            MaxClasses = MaxClasses[0]
            if MaxClasses < 2:
                MaxClasses = 2
            if MaxClasses >= len(Data):
                MaxClasses = len(Data)
            Nc2test = [MaxClasses]
        else:
            Nc2test = MaxClasses

    if N==1:
        return (1, [[0]])
    else:
        if Algo == "HC":
            ThresholdDendrogram = Dendrogram(Data=Data, methodHC=methodHC, distance=Distance, ThresholdDendrogram=ThresholdDendrogram)
            ClustersHC = HierarchicalClustering(Data=Data, methodHC=methodHC, distance=Distance, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=Nc2test, criterion = criterionHC,
                                                 DistanceStr = DistanceStr,
                                                 pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate = SamplingRate)
            NClasses = len(ClustersHC)
            Clusters = []
            for I in range(NClasses):
                if len(ClustersHC[I])>0:
                    Clusters.append(ClustersHC[I])
            NClasses = len(Clusters)

        elif Algo == "KM":
            Clusters, NClasses, Centers = Kmeans_Algo(Data=Data, nc2test=Nc2test, distance=Distance, methodKM=methodKM)


        elif Algo == "FCM":
            ClustersKM, NClassesKM, CentersKM = Kmeans_Algo(Data=Data, nc2test=Nc2test, distance=Distance, methodKM=methodKM)
            Clusters, Centers, MembershipMat = Fcm.FCM(Data=Data, NClasses = NClassesKM, Centers=CentersKM, FuzzyParameter=FuzzyParameter, MaxIter=MaxIterFCM, Metric=Distance)
            NClasses = len(Clusters)

        elif Algo == "KShape":
            Clusters, NClasses, Centers = Kshape_Algo(Data=Data, nc2test=Nc2test, methodKM=methodKM)

        elif Algo == "PCA&KShape":
            Data, DataPostPCA = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            Clusters, NClasses, Centers = Kshape_Algo(Data=Data, nc2test=Nc2test, methodKM=methodKM)

        elif Algo == "ICA&KShape":
            Data, DataPostICA = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            Clusters, NClasses, Centers = Kshape_Algo(Data=Data, nc2test=Nc2test, methodKM=methodKM)

        elif Algo == "KernelPCA&KShape":
            Data, DataPostPCA = kernelPCA_Algo(Data=Data, ncomp = kpca_ncomp)
            Clusters, NClasses, Centers = Kshape_Algo(Data=Data, nc2test=Nc2test, methodKM=methodKM)



        elif Algo=="Leiden":
            ClustersL = Leiden_Algo(Data=Data.T, ThresholdLeiden=ThresholdLeiden, DistanceStr=DistanceStr, pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate=SamplingRate)[0]
            NClasses = len(ClustersL)
            Clusters = []
            for I in range(NClasses):
                if len(ClustersL[I][0])>0:
                    Clusters.append(ClustersL[I][0])
            NClasses = len(Clusters)


        elif Algo=="PCA&HC":
            Data, DataPostPCA  = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            ThresholdDendrogram = Dendrogram(Data=Data, methodHC=methodHC, distance=Distance, ThresholdDendrogram=ThresholdDendrogram)
            ClustersHC = HierarchicalClustering(Data=Data, methodHC=methodHC, distance=Distance, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=Nc2test, criterion=criterionHC,
                                                 DistanceStr=DistanceStr,
                                                 pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate = SamplingRate)
            NClasses = len(ClustersHC)

            Clusters = []
            for I in range(NClasses):
                if len(ClustersHC[I])>0:
                    Clusters.append(ClustersHC[I])
            NClasses = len(Clusters)

        elif Algo == "PCA&KM":
            Data, DataPostPCA = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            Clusters, NClasses, Centers = Kmeans_Algo(Data=Data, nc2test=Nc2test, distance=Distance, methodKM=methodKM)



        elif Algo == "PCA&FCM":
            Data, DataPostPCA  = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            ClustersKM, NClassesKM, CentersKM = Kmeans_Algo(Data=Data, nc2test=Nc2test, distance=Distance, methodKM=methodKM)
            MaxIter = 5
            Clusters, CentersFCM, MembershipMat = Fcm.FCM(Data=Data, NClasses = NClassesKM, Centers=CentersKM, FuzzyParameter=FuzzyParameter, MaxIter=MaxIterFCM, Metric=Distance)
            NClasses = len(Clusters)


        elif Algo =="PCA&Leiden":
            Data, DataPostPCA  = PCA_Algo(Data=Data, ThresholdVariance=ThresholdVariance)
            ClustersL = Leiden_Algo(Data=Data.T, ThresholdLeiden=ThresholdLeiden, DistanceStr=DistanceStr, pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate=SamplingRate)[0]
            NClasses = len(ClustersL)
            Clusters = []
            for I in range(NClasses):
                if len(ClustersL[I][0])>0:
                    Clusters.append(ClustersL[I][0])
            NClasses = len(Clusters)

        elif Algo=="ICA&HC":
            Data, DataPostICA = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            ThresholdDendrogram = Dendrogram(Data=Data, methodHC=methodHC, distance=Distance, ThresholdDendrogram=ThresholdDendrogram)
            ClustersHC = HierarchicalClustering(Data=Data, methodHC=methodHC, distance=Distance, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=Nc2test, criterion=criterionHC,
                                                 DistanceStr = DistanceStr,
                                                 pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate = SamplingRate)
            NClasses = len(ClustersHC)
            Clusters = []
            for I in range(NClasses):
                if len(ClustersHC[I])>0:
                    Clusters.append(ClustersHC[I])
            NClasses = len(Clusters)


        elif Algo == "ICA&KM":
            Data, DataPostICA  = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            Clusters, NClasses, Centers = Kmeans_Algo(Data=Data, nc2test=Nc2test, distance=Distance, methodKM=methodKM)


        elif Algo == "ICA&FCM":
            Data, DataPostICA   = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            ClustersKM, NClassesKM, CentersKM = Kmeans_Algo(Data=Data, nc2test=Nc2test, distance=Distance, methodKM=methodKM)
            MaxIter = 5
            Clusters, CentersFCM, MembershipMat = Fcm.FCM(Data=Data, NClasses = NClassesKM, Centers=CentersKM, FuzzyParameter=FuzzyParameter, MaxIter=MaxIterFCM, Metric=Distance)
            NClasses = len(Clusters)


        elif Algo =="ICA&Leiden":
            Data, DataPostICA   = ICA_Algo(Data=Data, ncomp = ica_ncomp)
            ClustersL = Leiden_Algo(Data=Data.T, ThresholdLeiden=ThresholdLeiden, DistanceStr=DistanceStr, pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate=SamplingRate)[0]
            NClasses = len(ClustersL)
            Clusters = []
            for I in range(NClasses):
                if len(ClustersL[I][0])>0:
                    Clusters.append(ClustersL[I][0])
            NClasses = len(Clusters)

        elif Algo=="kernelPCA&HC":
            Data, DataPostPCA  = kernelPCA_Algo(Data=Data, ncomp = kpca_ncomp)
            ThresholdDendrogram = Dendrogram(Data=Data, methodHC=methodHC, distance=Distance, ThresholdDendrogram=ThresholdDendrogram)
            ClustersHC = HierarchicalClustering(Data=Data, methodHC=methodHC, distance=Distance, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=Nc2test, criterion=criterionHC,
                                                 DistanceStr = DistanceStr,
                                                 pMinkowski  = pMinkowski , wMax = wMax, g = g, epsilonEDR = epsilonEDR, epsilonLCSS = epsilonLCSS, SamplingRate = SamplingRate)
            NClasses = len(ClustersHC)
            Clusters = []
            for I in range(NClasses):
                if len(ClustersHC[I])>0:
                    Clusters.append(ClustersHC[I])
            NClasses = len(Clusters)


        elif Algo == "kernelPCA&KM":
            Data, DataPostPCA = kernelPCA_Algo(Data=Data, ncomp = kpca_ncomp)
            Clusters, NClasses, Centers = Kmeans_Algo(Data=Data, nc2test=Nc2test, distance=Distance, methodKM=methodKM)



        elif Algo == "kernelPCA&FCM":
            Data, DataPostPCA  = kernelPCA_Algo(Data=Data, ncomp = kpca_ncomp)
            ClustersKM, NClassesKM, CentersKM = Kmeans_Algo(Data=Data, nc2test=Nc2test, distance=Distance, methodKM=methodKM)
            MaxIter = 5
            Clusters, CentersFCM, MembershipMat = Fcm.FCM(Data=Data, NClasses = NClassesKM, Centers=CentersKM, FuzzyParameter=FuzzyParameter, MaxIter=MaxIterFCM, Metric=Distance)
            NClasses = len(Clusters)

        return (NClasses, Clusters)

def RecursiveClustering(Data, Algo = 'KM', DistanceStr = 'm', methodHC = 'complete', criterionHC = 'distance', methodKM = 'silhouette', MaxIterFCM=10, ThresholdVariance = 0.9, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, FuzzyParameter = 1, noise = 0, ThresholdDendrogram = 0.33, MaxClasses = [2], ThresholdLeiden = 0.9, SamplingRate = 1000, pMinkowski  = 2, Normalization = 'OFF', NormMode ='min_max_single'):
    """
    This Algorithm is recursive, based on the sum of squares criteria.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    Algo : str
        clustering Algorithm. Choices: c-means ('KM'), fuzzy c-means ('FCM'), hierarchical clustering ('HC'), Leiden ('Leiden') and applying first a dimensionality reduction by PCA ('PCA&KM', 'PCA&FCM', 'PCA&HC', 'PCA&Leiden'). Defaults to 'KM'.
    DistanceStr : str
        metric used for clustering. Defaults to 'm'.
    methodHC : str
        linkage method. Choices: 'complete', 'single', 'average'. Defaults to 'complete'.
    criterionHC : str
        Hierarchical clustering criterion. Choices: 'distance', 'maxclust'. Defaults to 'distance'.
    methodKM : str
        Method to selct the optimal number of centroid. 'Choices: 'silhouette', 'wcss'. Defaults to 'silhouette'.
    MaxIterFCM : int
        maximum number of iterations for FCM. Defaults to 10.
    ThresholdVariance : float
        explained variance after PCA. Defaults to 0.9.
    wMax : int
        WDTW and WDDTW parameter. Defaults to 1.
    g : int
        WDTW and WDDTW parameter. Defaults to 1.
    epsilonEDR : float
        EDR threshold. Defaults to 0.001.
    epsilonLCSS : float
        LCSS threshold. Defaults to 0.001.
    FuzzyParameter : int
        FCM parameter. Defaults to 1.
    noise : int
        Percentage of noise to add to Data. Defaults to 0.
    ThresholdDendrogram : float
        Cut height in percentage. Defaults to 0.7.
    MaxClasses : int
        maximum number of classes. Defaults to 1.
    ThresholdLeiden : float
        Leiden threshold. Defaults to 0.9.
    SamplingRate : int
        STS parameter. Defaults to 1000.
    pMinkowski : int
        Minkowski parameter. Defaults to 2.
    Normalization : str
        To applying Normalization. Choices: 'ON', 'OFF'. Defaults to 'OFF'.
    NormMode : str
        If Normalization applied, to select the modality. Choices: 'min_max_single', 'min_max_global', 'mu_std_single', 'mu_std_global'. Defaults to 'min_max_single'.
    """

    Distance = 'd_'+DistanceStr
    Distance = globals()[Distance]

    DataPlot = Data.copy()

    Distance.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
    Media = np.mean(DataPlot,0)
    SS = 0
    for I in range(DataPlot.shape[0]):
        SS += Distance(DataPlot[I], Media)**2
    Clusters = Clustering(Data=Data, Algo=Algo, DistanceStr=DistanceStr, methodHC = methodHC, criterionHC=criterionHC, methodKM=methodKM, MaxIterFCM=MaxIterFCM, ThresholdVariance=ThresholdVariance, wMax=wMax, g=g, epsilonEDR=epsilonEDR, epsilonLCSS=epsilonLCSS, FuzzyParameter=FuzzyParameter, noise=noise, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=MaxClasses, ThresholdLeiden=ThresholdLeiden, SamplingRate=SamplingRate, pMinkowski =pMinkowski , Normalization=Normalization, NormMode=NormMode)
    Clusters = Clusters[1]
    WcsskList = []
    WcssValue = 0
    for J in range(len(Clusters)):
        Wcssk = 0
        ClusterPoints = DataPlot[Clusters[J]]
        ClusterPoints=ClusterPoints.reshape((ClusterPoints.shape[-2],ClusterPoints.shape[-1]))
        Center = np.mean(ClusterPoints,0)
        for I in range(ClusterPoints.shape[0]):
            Wcssk += Distance(ClusterPoints[I], Center)**2
        WcssValue+=Wcssk
        WcsskList.append(Wcssk)
    while max(WcsskList)>SS/100*25:
        NewClusters = []
        for J in range(len(Clusters)):
            if WcsskList[J]>SS/100*25:
                Data = DataPlot.copy()
                ClustersJ = Clustering(Data=Data[Clusters[J]].reshape(Data[Clusters[J]].shape[-2],Data[Clusters[J]].shape[-1]), Algo=Algo, DistanceStr=DistanceStr, methodHC = methodHC, criterionHC=criterionHC, methodKM=methodKM, MaxIterFCM=MaxIterFCM, ThresholdVariance=ThresholdVariance, wMax=wMax, g=g, epsilonEDR=epsilonEDR, epsilonLCSS=epsilonLCSS, FuzzyParameter=FuzzyParameter, noise=noise, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=MaxClasses, ThresholdLeiden=ThresholdLeiden, SamplingRate=SamplingRate, pMinkowski =pMinkowski , Normalization=Normalization, NormMode=NormMode)
                ClustersJ = ClustersJ[1]
                Indexes = np.array(sorted(Clusters[J]))
                for I in range(len(ClustersJ)):
                    NewClusters.append(list(Indexes[ClustersJ[I]]))
            else:
                NewClusters.append(Clusters[J])
        Clusters = NewClusters
        WcsskList = []
        WcssValue = 0
        for J in range(len(Clusters)):
            Wcssk = 0
            ClusterPoints = DataPlot[Clusters[J]]
            ClusterPoints=ClusterPoints.reshape((ClusterPoints.shape[-2],ClusterPoints.shape[-1]))
            Center = np.mean(ClusterPoints,0)
            for I in range(ClusterPoints.shape[0]):
                Wcssk += Distance(ClusterPoints[I], Center)**2
            WcssValue+=Wcssk
            WcsskList.append(Wcssk)

    return (len(Clusters),Clusters)

def ClusterCentroids(Data, Clusters):
    """
    Given a Dataset and its subdivision in Clusters, the function returns the Centers of the Clusters.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    Clusters : tuple
        number of Clusters and Clusters obtained by the clustering Algorithm applied.

    Returns
    -------
    Centroids : np.ndarray
        a vector with the coordinates of the Centers of the Clusters.
    """
    NClasses = len(Clusters[1])
    Centroids = np.zeros((NClasses, Data.shape[1]))
    for I in range(NClasses):
        Centroids[I]=np.mean(Data[Clusters[1][I]],0)
    return Centroids

def Classification(Centroids, Data, DistanceStr='m', pMinkowski  = 2, wMax = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, SamplingRate=1000):
    """
    Given a Dataset, a group of Centroids and a metric to compute distances between each Data and each Centroids,.

    Parameters
    ----------
    Centroids : np.ndarray
        a vector with the coordinates of the Centers of some Clusters.
    Data : np.ndarray
        2D matrix representing the Dataset.
    DistanceStr : str
        metric to compute distances. Defaults to 'ed'.

    Returns
    -------
    Classification : list
        Clusters obtained from the Dataset assigning each Data to the class of the closest centroid.
    """
    Distance = 'd_'+DistanceStr
    Metric = globals()[Distance]
    Metric.__defaults__ = (pMinkowski , wMax, g, epsilonEDR, epsilonLCSS, SamplingRate)
    N = Data.shape[0]
    X = Data.shape[1]
    C = Centroids.shape[0]
    Classification = []
    for I in range(N):
        Distances = []
        for J in range(C):
            Distances.append(Metric(Data[I],Centroids[J]))
        Distances = np.array(Distances)
        IdxM = np.where(Distances==min(Distances))[0][0]
        IdxM = int(IdxM)
        Classification.append(IdxM)

    return Classification

def GaussianNoise(Data, noise, seed):
    """
    Function to add Gaussian noise to Data.

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Dataset.
    noise : float
        amount of noise to add.
    seed : int
        seed for random generation.

    Returns
    -------
    Data : np.ndarray
        2D matrix representing the Dataset after the adding of noise.
    """
    np.random.seed(seed)
    for I in range(len(Data)):
        Sigma = np.std(Data[I])
        if Sigma == 0:
            Data[I] = Data[I]+noise*np.random.normal(0,1,len(Data[I]))
        else:
            Data[I] = Data[I]+noise*np.random.normal(0,Sigma,len(Data[I]))

    return Data
