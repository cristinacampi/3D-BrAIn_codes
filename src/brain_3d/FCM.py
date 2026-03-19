"""Fuzzy C-Means clustering implementation."""

import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt


def InitializeMembershipMatrix(n_points, n_classes):
    """Initialize fuzzy membership matrix with random values.
    
    Creates a matrix where each row represents a data point and each column 
    represents membership to a cluster. Values are normalized to sum to 1.
    
    Args:
        n_points (int): number of data points
        n_classes (int): number of clusters
    
    Returns:
        list: membership matrix of shape [n_points, n_classes] with random normalized values
    """
    membership_mat = []
    for i in range(n_points):
        # Generate random membership values
        random_num_list = [random.random() for _ in range(n_classes)]
        summation = sum(random_num_list)
        
        # Normalize to sum to 1
        temp_list = [x / summation for x in random_num_list]
        
        # Set max value to 1, others to 0 (hard initialization)
        flag = temp_list.index(max(temp_list))
        for j in range(len(temp_list)):
            temp_list[j] = 1 if j == flag else 0
        
        membership_mat.append(temp_list)
    
    return membership_mat


def CalculateClusterCenters(data, membership_mat, n_points, n_classes, fuzzy_parameter):
    """Calculate cluster centers based on membership matrix.
    
    Args:
        data (array): dataset of shape [n_points, n_features]
        membership_mat (list): membership matrix of shape [n_points, n_classes]
        n_points (int): number of data points
        n_classes (int): number of clusters
        fuzzy_parameter (float): fuzziness parameter (m > 1, typically 2)
    
    Returns:
        list: cluster centers of shape [n_classes, n_features]
    """
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = []
    
    for j in range(n_classes):
        x = list(cluster_mem_val[j])
        xraised = [p ** fuzzy_parameter for p in x]
        denominator = sum(xraised)
        
        # Calculate weighted sum of data points
        temp_num = []
        for i in range(n_points):
            data_point = list(data[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        
        numerator = list(map(sum, list(zip(*temp_num))))
        center = [z / denominator for z in numerator]
        cluster_centers.append(center)
    
    return cluster_centers


def UpdateMembershipValue(data, membership_mat, n_points, n_classes, fuzzy_parameter, cluster_centers, metric):
    """Update membership matrix based on distances to cluster centers.
    
    Args:
        data (array): dataset of shape [n_points, n_features]
        membership_mat (list): current membership matrix
        n_points (int): number of data points
        n_classes (int): number of clusters
        fuzzy_parameter (float): fuzziness parameter
        cluster_centers (list): cluster centers
        metric (function): distance metric function
    
    Returns:
        list: updated membership matrix
    """
    p = float(2 / (fuzzy_parameter - 1))
    
    for i in range(n_points):
        x = list(data[i])
        distances = []
        
        # Calculate distances to all cluster centers
        for k in range(n_classes):
            try:
                d = metric(x, cluster_centers[k])
            except Exception:
                d = 0
            distances.append(d)
        
        # Check for zero distances (data point coincides with cluster center)
        idx = np.where(np.array(distances) == 0)[0]
        
        if len(idx) == 0:
            # Update membership values using fuzzy formula
            for j in range(n_classes):
                den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(n_classes)])
                membership_mat[i][j] = float(1 / den)
        else:
            # Point coincides with cluster center: membership = 1 for that cluster
            for j in range(n_classes):
                membership_mat[i][j] = 0
            membership_mat[i][idx[0]] = 1
    
    return membership_mat


def GetClusters(membership_mat, n_points):
    """Extract cluster labels from membership matrix.
    
    Assigns each point to the cluster with maximum membership value.
    
    Args:
        membership_mat (list): membership matrix of shape [n_points, n_classes]
        n_points (int): number of data points
    
    Returns:
        list: cluster labels for each data point
    """
    cluster_labels = []
    for i in range(n_points):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    
    return cluster_labels


def FuzzyCMeansClustering(data, n_points, n_classes, centers, fuzzy_parameter, max_iter, metric):
    """Perform Fuzzy C-Means clustering algorithm.
    
    Args:
        data (array): dataset of shape [n_points, n_features]
        n_points (int): number of data points
        n_classes (int): number of clusters
        centers (list): initial cluster centers
        fuzzy_parameter (float): fuzziness parameter (m > 1)
        max_iter (int): maximum number of iterations
        metric (function): distance metric function
    
    Returns:
        tuple: (cluster_labels, cluster_centers, iteration_history, membership_matrix)
            - cluster_labels (list): final cluster assignment for each point
            - cluster_centers (list): final cluster centers
            - iteration_history (list): cluster labels at each iteration
            - membership_mat (list): final membership matrix
    """
    # Initialize membership matrix
    membership_mat = InitializeMembershipMatrix(n_points, n_classes)
    curr = 0
    acc = []
    cent_temp = centers
    
    while curr < max_iter:
        if curr == 0:
            cluster_centers = cent_temp
        else:
            cluster_centers = CalculateClusterCenters(data, membership_mat, n_points, n_classes, fuzzy_parameter)
        
        # Update memberships and get labels
        membership_mat = UpdateMembershipValue(data, membership_mat, n_points, n_classes, 
                                             fuzzy_parameter, cluster_centers, metric)
        cluster_labels = GetClusters(membership_mat, n_points)
        acc.append(cluster_labels)
        curr += 1
    
    return cluster_labels, cluster_centers, acc, membership_mat


def FCM(data, n_classes, centers, fuzzy_parameter, max_iter, metric):
    """Fuzzy C-Means clustering wrapper function.
    
    Performs fuzzy clustering and returns cluster assignments and centers.
    
    Args:
        data (array): dataset of shape [n_points, n_features]
        n_classes (int): number of clusters
        centers (list): initial cluster centers of shape [n_classes, n_features]
        fuzzy_parameter (float): fuzziness parameter (m > 1, typically 2)
        max_iter (int): maximum number of iterations
        metric (function): distance metric function taking (point1, point2) as arguments
    
    Returns:
        tuple: (clusters, centers, membership_matrix)
            - clusters (list): list of arrays, each containing indices of points in that cluster
            - centers (list): final cluster centers
            - membership_matrix (list): final fuzzy membership matrix
    """
    n_points = len(data)
    
    # Run FCM algorithm
    labels, centers, acc, membership_mat = FuzzyCMeansClustering(
        data, n_points, n_classes, centers, fuzzy_parameter, max_iter, metric
    )
    
    # Convert labels to cluster index lists
    clusters = []
    labels = np.array(labels)
    for i in range(n_classes):
        indexes = np.where(labels == i)[0]
        clusters.append(indexes)
    
    return clusters, centers, membership_mat