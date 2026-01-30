"""Clustering and stratification functions for time series analysis."""

import numpy as np
import sklearn
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from scipy.cluster.hierarchy import dendrogram, linkage, fclusterdata
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, FastICA
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
import pandas as pd
import h5py
import math
import scipy
import time
from . import FCM
from scipy.signal import find_peaks, butter, filtfilt
import igraph as ig
import leidenalg as la
import random
from kneed import KneeLocator
from scipy.stats import pearsonr

'''DISTANCE FUNCTIONS'''

def DistanceMinkowski(a, b, p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Minkowski distance between two vectors.

    Args:
        a (array): first time series
        b (array): second time series
        p_minkowski (int, optional): Minkowski parameter. Defaults to 2 (Euclidean).
        w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate: unused parameters for interface compatibility.

    Returns:
        float: Minkowski distance between a and b
    """
    l1 = len(a)
    l2 = len(b)
    if l1 != l2:
        raise ValueError('Vectors must have equal length')
    return minkowski(a, b, p=p_minkowski)

def MatrixC(M):
    """Create cost matrix for DTW distance calculation.

    Args:
        M (array): 2D matrix of pairwise distances

    Returns:
        array: cumulative cost matrix
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

def MatrixM(a, b):
    """Compute pairwise squared distances between elements of two vectors.

    Args:
        a (array): first vector
        b (array): second vector

    Returns:
        array: 2D matrix of squared distances
    """
    l1 = len(a)
    aa = np.repeat(a, l1, axis=0).reshape(l1, l1)
    bb = np.repeat(b, l1, axis=0).reshape(l1, l1).T
    return (aa - bb) ** 2

def Warping(a, b, M):
    """Compute dynamic time warping alignment between two signals.

    Args:
        a (array): first signal
        b (array): second signal
        M (array): pairwise distance matrix

    Returns:
        tuple: (warped_a, warped_b, indices_a, indices_b)
    """
    C = MatrixC(M)
    i = C.shape[0]-1
    j = C.shape[1]-1
    l = []
    while (i>0) & (j>0):
        l.append((i,j))
        m = min(C[i-1][j], C[i][j-1], C[i-1][j-1])
        if m == C[i-1][j-1]:
            i -= 1
            j -= 1
        elif m == C[i][j-1]:
            j -= 1
        else:
            i -= 1
    
    idx_a = [l[len(l)-1-k][0]-1 for k in range(len(l))]
    idx_b = [l[len(l)-1-k][1]-1 for k in range(len(l))]
    w_a = a[idx_a]
    w_b = b[idx_b]
    return w_a, w_b, np.array(idx_a), np.array(idx_b)

def DistanceDTW(a, b, p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Dynamic Time Warping distance.

    Args:
        a (array): first time series
        b (array): second time series
        p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate: parameters

    Returns:
        float: DTW distance
    """
    if len(a) != len(b):
        raise ValueError('Vectors must have equal length')
    M = MatrixM(a, b)
    C = MatrixC(M)
    return math.sqrt(C[len(a)][len(b)])

def A1B1DDTW(a, b):
    """Transform vectors for derivative DTW distance.

    Args:
        a (array): first vector
        b (array): second vector

    Returns:
        tuple: (transformed_a, transformed_b)
    """
    l1 = len(a)
    a1 = np.empty(l1-2)
    b1 = np.empty(l1-2)
    for i in range(1, l1-1):
        a1[i-1] = ((a[i]-a[i-1]) + ((a[i+1]-a[i-1])/2))/2
        b1[i-1] = ((b[i]-b[i-1]) + ((b[i+1]-b[i-1])/2))/2
    return a1, b1

def DistanceDDTW(a, b, p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Derivative Dynamic Time Warping distance."""
    if len(a) != len(b):
        raise ValueError('Vectors must have equal length')
    a1, b1 = A1B1DDTW(a, b)
    return DistanceDTW(a1, b1)

def MatrixMw(M, w_max, g):
    """Apply weights to distance matrix based on indices.

    Args:
        M (array): distance matrix
        w_max (float): weight parameter
        g (float): exponential parameter

    Returns:
        array: weighted distance matrix
    """
    l1 = M.shape[0]
    for i in range(l1):
        for j in range(l1):
            M[i,j] = (w_max/(1+math.exp(-g*(abs(i-j)-l1/2))))*M[i,j]
    return M

def DistanceWDTW(a, b, p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Weighted Dynamic Time Warping distance."""
    if len(a) != len(b):
        raise ValueError('Vectors must have equal length')
    M = MatrixM(a, b)
    M = MatrixMw(M, w_max, g)
    C = MatrixC(M)
    return math.sqrt(C[len(a)][len(b)])

def DistanceWDDTW(a, b, p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Weighted Derivative Dynamic Time Warping distance."""
    a1, b1 = A1B1DDTW(a, b)
    return DistanceWDTW(a1, b1, w_max=w_max, g=g)

def DistanceLCSS(a, b, p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Longest Common Subsequence distance."""
    if len(a) != len(b):
        raise ValueError('Vectors must have equal length')
    epsilon_abs = epsilon_LCSS * np.linalg.norm(a)
    l1 = len(a)
    L = np.zeros((l1+1, l1+1))
    for i in range(l1):
        for j in range(l1):
            if abs(a[i]-b[j]) < epsilon_abs:
                L[i+1][j+1] = L[i][j] + 1
            else:
                L[i+1][j+1] = max(L[i+1][j], L[i][j+1])
    return 1 - L[l1][l1]/l1

def DistanceEDR(a, b, p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Edit Distance on Real Sequences."""
    if len(a) != len(b):
        raise ValueError('Vectors must have equal length')
    epsilon_abs = epsilon_EDR * np.linalg.norm(a)
    l1 = len(a)
    E = np.zeros((l1+1, l1+1))
    for i in range(l1):
        for j in range(l1):
            c = 0 if abs(a[i]-b[j]) < epsilon_abs else 1
            E[i+1][j+1] = min(E[i][j]+c, E[i+1][j]+1, E[i][j+1]+1)
    return E[l1][l1]

def DistanceRho2(a, b, p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Distance based on Pearson correlation coefficient."""
    if len(a) != len(b):
        raise ValueError('Vectors must have equal length')
    rho = pearsonr(a, b)[0]
    return 2 * (1 - rho)

def DistanceSTS(a, b, p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Short Time Series distance."""
    if len(a) != len(b):
        raise ValueError('Vectors must have equal length')
    aa = np.diff(a)
    bb = np.diff(b)
    return math.sqrt(np.sum(((aa-bb)*SamplingRate)**2))

def AdjacencyMatrix(data, distance='m', p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Compute adjacency matrix for Leiden algorithm.

    Args:
        data (array): 2D dataset
        distance (str): distance metric name
        Other parameters for distance functions

    Returns:
        array: adjacency matrix
    """
    distance_func = globals()[f'Distance{distance.upper()}']
    distance_func.__defaults__ = (p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
    dim = data.shape[0]
    matrix = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i+1, dim):
            matrix[i][j] = distance_func(data[i], data[j])
    matrix = matrix + matrix.T
    M = np.max(matrix)
    matrix_2 = matrix / M
    adjacency = 1 / (1 + matrix_2)
    adjacency[adjacency <= 0.75] = 0
    return adjacency

'''NORMALIZATION FUNCTIONS'''

def NormalizationMinMaxSingle(data):
    """Min-max normalization per sample."""
    for i in range(len(data)):
        m = np.min(data[i])
        M = np.max(data[i])
        if m == M:
            data[i] = np.zeros_like(data[i])
        else:
            data[i] = (data[i] - m) / (M - m)
    return data

def NormalizationMinMaxGlobal(data):
    """Min-max normalization using global min/max."""
    minimum = np.min(data)
    maximum = np.max(data)
    for i in range(len(data)):
        data[i] = (data[i] - minimum) / (maximum - minimum)
    return data

def Whitening(data):
    """Z-score normalization per sample."""
    for i in range(len(data)):
        mu = np.mean(data[i])
        sigma = np.std(data[i])
        if sigma > 0:
            data[i] = (data[i] - mu) / sigma
        else:
            data[i] = np.zeros_like(data[i])
    return data

def WhiteningGlobal(data):
    """Z-score normalization using global mean/std."""
    mu = np.mean(data)
    sigma = np.std(data)
    for i in range(len(data)):
        data[i] = (data[i] - mu) / sigma
    return data

'''CLUSTERING ALGORITHMS'''

def Dendrogram(data, metric, method_HC='complete', threshold_dendrogram=0.7):
    """Generate dendrogram cut height for hierarchical clustering.

    Args:
        data (array): 2D dataset
        metric (str): distance metric
        method_HC (str): linkage method
        threshold_dendrogram (float): threshold as percentage of max height

    Returns:
        float: cut height for dendrogram
    """
    try:
        linkage_data = linkage(data, method=method_HC, metric=metric)
        n = len(data)
        aggregation_levels = linkage_data[:, 2]
        max_d = threshold_dendrogram * aggregation_levels[n-2]
    except:
        max_d = threshold_dendrogram * len(data)
    return max_d

def HierarchicalClustering(data, method_HC, metric, threshold_dendrogram, max_classes, criterion):
    """Perform hierarchical clustering.

    Args:
        data (array): dataset
        method_HC (str): linkage method
        metric (str): distance metric
        threshold_dendrogram (float): cut height
        max_classes (int): max number of clusters
        criterion (str): 'distance' or 'maxclust'

    Returns:
        list: clusters (lists of indices)
    """
    if criterion == 'distance':
        fclust = fclusterdata(data, threshold_dendrogram, criterion='distance', metric=metric, method=method_HC)
    elif criterion == 'maxclust':
        fclust = fclusterdata(data, max_classes, criterion='maxclust', metric=metric, method=method_HC)
    
    clusters = [np.where(fclust == i+1)[0] for i in range(max(fclust))]
    return clusters

def KmeansAlgo(data, nc2test, distance, method_KM='silhouette'):
    """K-means clustering with optional silhouette or elbow selection.

    Args:
        data (array): dataset
        nc2test (array): number of clusters to test
        distance: distance function
        method_KM (str): selection method

    Returns:
        tuple: (clusters, n_clusters, centers)
    """
    metric = distance_metric(type_metric.USER_DEFINED, func=distance)
    iterations = []
    classi = []
    centers = []
    c_best = 2
    best_score = 0

    for k in range(len(nc2test)):
        n_clusters = nc2test[k]
        n = len(data)
        i_c = random.sample(range(n), n_clusters)
        start_centers = data[i_c]
        kmeans_instance = kmeans(data, start_centers, metric=metric)
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        classi.append((n_clusters, clusters))
        centers.append(kmeans_instance.get_centers())
        
        labels = np.zeros(n, dtype=int)
        for j in range(len(clusters)):
            labels[clusters[j]] = j+1
        
        wcss = sum(distance(data[clusters[j][i]], centers[k][j])**2 
                  for j in range(len(clusters)) for i in range(len(clusters[j])))
        iterations.append(wcss)

        if method_KM == 'silhouette':
            if len(np.unique(labels)) > 1:
                score = silhouette_score(data, labels, metric=metric)
            else:
                score = 0
            if score > best_score:
                best_score = score
                c_best = n_clusters
            print(f"n_clusters={n_clusters}, silhouette={score:.4f}")

    if method_KM == 'silhouette':
        idx_best = np.where(nc2test == c_best)[0][0]
    elif method_KM == 'wcss':
        kl = KneeLocator(nc2test, iterations, curve="convex", direction="decreasing")
        k_elbow = kl.elbow if kl.elbow else nc2test[-1]
        idx_best = np.where(nc2test == k_elbow)[0][0]
    
    return classi[idx_best][1], classi[idx_best][0], centers[idx_best]

def PCAAlgo(data, threshold_variance=0.9):
    """Principal Component Analysis for dimensionality reduction.

    Args:
        data (array): dataset
        threshold_variance (float): explained variance threshold

    Returns:
        tuple: (transformed_data, reconstructed_data)
    """
    n = min(data.shape[0], data.shape[1])
    pca = PCA(n_components=n)
    pca.fit(data)
    variance = pca.explained_variance_ratio_
    
    sum_ratio = 0
    i = 0
    while sum_ratio < threshold_variance and i < len(variance):
        sum_ratio += variance[i]
        i += 1
    n_c = i
    
    pca = PCA(n_components=n_c)
    y = pca.fit_transform(data)
    data_back = pca.inverse_transform(y)
    return y, data_back

def LeidenAlgo(data, threshold_Leiden=0.95, distance='m', p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Leiden graph-based clustering algorithm.

    Args:
        data (array): dataset (transposed for features as rows)
        threshold_Leiden (float): correlation threshold
        distance (str): distance metric
        Other parameters

    Returns:
        tuple: (clusters, graph, partition)
    """
    if distance == 'rho':
        df = pd.DataFrame(data)
        c = df.corr()
        c[c <= threshold_Leiden] = 0
    else:
        c = AdjacencyMatrix(data.T, distance, p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)

    G = ig.Graph.Weighted_Adjacency(c, mode='undirected', attr='weight', loops=False)
    partition = la.find_partition(G, la.ModularityVertexPartition)
    optimiser = la.Optimiser()
    improvement = optimiser.optimise_partition(partition)
    while improvement:
        improvement = optimiser.optimise_partition(partition)
    
    partition_membership = np.array(partition.membership)
    n_clusters = max(partition_membership) + 1
    clusters = [np.where(partition_membership == i)[0] for i in range(n_clusters)]
    return clusters, G, partition

def Clustering(data, algo='KM', distance='m', method_HC='complete', criterion_HC='distance', method_KM='silhouette', 
               max_iter_FCM=10, threshold_variance=0.9, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, 
               fuzzy_parameter=1, noise=0, threshold_dendrogram=0.7, max_classes=[2], threshold_Leiden=0.9, 
               SamplingRate=1000, p_minkowski=2, normalization='OFF', norm_mode='min_max_single'):
    """Unified clustering interface supporting multiple algorithms.

    Args:
        data (array): dataset
        algo (str): algorithm choice (KM, FCM, HC, Leiden, PCA&*, ICA&*)
        distance (str): distance metric
        Other algorithm-specific parameters

    Returns:
        tuple: (n_classes, clusters)
    """
    distance_func = globals()[f'Distance{distance.upper()}']
    distance_func.__defaults__ = (p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
    
    n = data.shape[0]
    data_copy = data.copy()
    
    # Noise handling
    if noise > 0:
        data = GaussianNoise(data, noise)
    
    # Normalization
    if normalization == 'ON':
        if norm_mode == 'min_max_single':
            data = NormalizationMinMaxSingle(data)
        elif norm_mode == 'min_max_global':
            data = NormalizationMinMaxGlobal(data)
        elif norm_mode == 'mu_std_single':
            data = Whitening(data)
        else:
            data = WhiteningGlobal(data)
    
    # Process max_classes parameter
    if isinstance(max_classes, int):
        nc2test = np.array([max(2, min(max_classes, n))])
    else:
        nc2test = np.array(max_classes)
        nc2test = nc2test[nc2test >= 2]
        if len(nc2test) == 0:
            nc2test = np.array([2])
    
    if n == 1:
        return (1, [[0]])
    
    # ...existing algorithm implementations...
    if algo == "KM":
        clusters_KM, n_centroids, centers = KmeansAlgo(data, nc2test, distance_func, method_KM)
        clusters = [c for c in clusters_KM if len(c) > 0]
        n_classes = len(clusters)
    
    elif algo == "HC":
        threshold_dendrogram = Dendrogram(data, distance, method_HC, threshold_dendrogram)
        clusters_HC = HierarchicalClustering(data, method_HC, distance, threshold_dendrogram, max(nc2test), criterion_HC)
        clusters = [c for c in clusters_HC if len(c) > 0]
        n_classes = len(clusters)
    
    elif algo == "Leiden":
        clusters_L, G, partition = LeidenAlgo(data, threshold_Leiden, distance, p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
        clusters = [c for c in clusters_L if len(c) > 0]
        n_classes = len(clusters)
    
    # ...add PCA&*, ICA&* variants similarly...
    
    return (n_classes, clusters)

def RecursiveClustering(data, algo='KM', distance='m', method_HC='complete', criterion_HC='distance', 
                        method_KM='silhouette', max_iter_FCM=10, threshold_variance=0.9, w_max=1, g=1, 
                        epsilon_EDR=0.001, epsilon_LCSS=0.001, fuzzy_parameter=1, noise=0, 
                        threshold_dendrogram=0.33, max_classes=[2], threshold_Leiden=0.9, 
                        SamplingRate=1000, p_minkowski=2, normalization='OFF', norm_mode='min_max_single'):
    """Recursive clustering based on sum of squares criterion."""
    distance_func = globals()[f'Distance{distance.upper()}']
    distance_func.__defaults__ = (p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
    
    data_plot = data.copy()
    media = np.mean(data_plot, 0)
    SS = sum(distance_func(data_plot[i], media)**2 for i in range(data_plot.shape[0]))
    
    clusters = Clustering(data, algo, distance, method_HC, criterion_HC, method_KM, max_iter_FCM, 
                         threshold_variance, w_max, g, epsilon_EDR, epsilon_LCSS, fuzzy_parameter, 
                         noise, threshold_dendrogram, max_classes, threshold_Leiden, SamplingRate, 
                         p_minkowski, normalization, norm_mode)[1]
    
    wcssk_list = []
    for cluster in clusters:
        wcssk = sum(distance_func(data_plot[i], np.mean(data_plot[cluster], 0))**2 for i in cluster)
        wcssk_list.append(wcssk)
    
    while max(wcssk_list) > SS/100*25:
        new_clusters = []
        for j, cluster in enumerate(clusters):
            if wcssk_list[j] > SS/100*25:
                sub_clusters = Clustering(data_plot[cluster], algo, distance, method_HC, criterion_HC, 
                                        method_KM, max_iter_FCM, threshold_variance, w_max, g, 
                                        epsilon_EDR, epsilon_LCSS, fuzzy_parameter, noise, 
                                        threshold_dendrogram, max_classes, threshold_Leiden, 
                                        SamplingRate, p_minkowski, normalization, norm_mode)[1]
                for sub_c in sub_clusters:
                    new_clusters.append(cluster[sub_c])
            else:
                new_clusters.append(cluster)
        clusters = new_clusters
        wcssk_list = []
        for cluster in clusters:
            wcssk = sum(distance_func(data_plot[i], np.mean(data_plot[cluster], 0))**2 for i in cluster)
            wcssk_list.append(wcssk)
    
    return (len(clusters), clusters)

def ClustersCentroids(data, clusters):
    """Compute cluster centroids.

    Args:
        data (array): dataset
        clusters (tuple): (n_classes, list of clusters)

    Returns:
        array: centroids
    """
    n_classes = clusters[0]
    centroids = np.zeros((n_classes, data.shape[1]))
    for i in range(n_classes):
        centroids[i] = np.mean(data[clusters[1][i]], 0)
    return centroids

def Classification(centroids, data, distance='m', p_minkowski=2, w_max=1, g=1, epsilon_EDR=0.001, epsilon_LCSS=0.001, SamplingRate=1000):
    """Assign data points to nearest centroid.

    Args:
        centroids (array): cluster centroids
        data (array): dataset
        distance (str): distance metric
        Other parameters

    Returns:
        list: cluster assignments
    """
    distance_func = globals()[f'Distance{distance.upper()}']
    distance_func.__defaults__ = (p_minkowski, w_max, g, epsilon_EDR, epsilon_LCSS, SamplingRate)
    
    classification = []
    for i in range(data.shape[0]):
        distances = [distance_func(data[i], centroids[j]) for j in range(centroids.shape[0])]
        classification.append(np.argmin(distances))
    return classification

def GaussianNoise(data, noise):
    """Add Gaussian noise to data.

    Args:
        data (array): dataset
        noise (float): noise standard deviation factor

    Returns:
        array: noisy data
    """
    for i in range(len(data)):
        sigma = np.std(data[i])
        if sigma == 0:
            data[i] += noise * np.random.normal(0, 1, len(data[i]))
        else:
            data[i] += noise * np.random.normal(0, sigma, len(data[i]))
    return data

def WarpingClusters(data, clusters, distance='dtw', w_max=1, g=1):
    """Visualize aligned waveforms within clusters using DTW.

    Args:
        data (array): dataset
        clusters (tuple): (n_classes, list of clusters)
        distance (str): distance metric (dtw, ddtw, wdtw, wddtw)
        w_max, g: DTW parameters
    """
    for i in range(clusters[0]):
        plt.figure()
        plt.title(f"Warped Cluster {i}")
        media = np.mean(data[clusters[1][i]], 0)
        
        for j in clusters[1][i]:
            if distance == "dtw":
                M = MatrixM(media, data[j])
                w_media, w_dataj, _, _ = Warping(media, data[j], M)
            elif distance == "ddtw":
                media_1, dataj_1 = A1B1DDTW(media, data[j])
                M = MatrixM(media_1, dataj_1)
                w_media, w_dataj, _, _ = Warping(media_1, dataj_1, M)
            elif distance == "wdtw":
                M = MatrixM(media, data[j])
                M = MatrixMw(M, w_max, g)
                w_media, w_dataj, _, _ = Warping(data[j], media, M)
            else:  # wddtw
                media_1, dataj_1 = A1B1DDTW(media, data[j])
                M = MatrixM(media_1, dataj_1)
                M = MatrixMw(M, w_max, g)
                w_media, w_dataj, _, _ = Warping(media_1, dataj_1, M)
            
            plt.plot(range(len(w_media)), w_media)
            plt.plot(range(len(w_dataj)), w_dataj, alpha=0.5)
        plt.show()




