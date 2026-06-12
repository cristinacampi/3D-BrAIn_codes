"""Fuzzy C-Means clustering implementation."""

import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt


def InitializeMembershipMatrix(NPoints, NClasses):
    """Initialize fuzzy membership matrix with random values.
    
    Creates a matrix where each row represents a Data point and each column 
    represents membership to a cluster. Values are normalized to sum to 1.
    
    Args:
        NPoints (int): number of Data points
        NClasses (int): number of clusters
    
    Returns:
        list: membership matrix of shape [NPoints, NClasses] with random normalized values
    """
    MembershipMat = []
    for i in range(NPoints):
        # Generate random membership values
        RandomNumList = [random.random() for _ in range(NClasses)]
        summation = sum(RandomNumList)
        
        # Normalize to sum to 1
        TempList = [x / summation for x in RandomNumList]
        
        # Set max value to 1, others to 0 (hard initialization)
        flag = TempList.index(max(TempList))
        for j in range(len(TempList)):
            TempList[j] = 1 if j == flag else 0
        
        MembershipMat.append(TempList)
    
    return MembershipMat


def CalculateClusterCenters(Data, MembershipMat, NPoints, NClasses, FuzzyParameter):
    """Calculate cluster centers based on membership matrix.
    
    Args:
        Data (array): Dataset of shape [NPoints, NFeatures]
        MembershipMat (list): membership matrix of shape [NPoints, NClasses]
        NPoints (int): number of Data points
        NClasses (int): number of clusters
        FuzzyParameter (float): fuzziness parameter (m > 1, typically 2)
    
    Returns:
        list: cluster centers of shape [NClasses, NFeatures]
    """
    ClusterMemVal = list(zip(*MembershipMat))
    ClusterCenters = []
    
    for j in range(NClasses):
        x = list(ClusterMemVal[j])
        xRaised = [p ** FuzzyParameter for p in x]
        Denominator = sum(xRaised)
        
        # Calculate weighted sum of Data points
        TempNum = []
        for i in range(NPoints):
            DataPoint = list(Data[i])
            prod = [xRaised[i] * val for val in DataPoint]
            TempNum.append(prod)
        
        Numerator = list(map(sum, list(zip(*TempNum))))
        Center = [z / Denominator for z in Numerator]
        ClusterCenters.append(Center)
    
    return ClusterCenters


def UpdateMembershipValue(Data, MembershipMat, NPoints, NClasses, FuzzyParameter, ClusterCenters, metric):
    """Update membership matrix based on distances to cluster centers.
    
    Args:
        Data (array): Dataset of shape [NPoints, NFeatures]
        MembershipMat (list): current membership matrix
        NPoints (int): number of Data points
        NClasses (int): number of clusters
        FuzzyParameter (float): fuzziness parameter
        ClusterCenters (list): cluster centers
        metric (function): distance metric function
    
    Returns:
        list: updated membership matrix
    """
    p = float(2 / (FuzzyParameter - 1))
    
    for i in range(NPoints):
        x = list(Data[i])
        distances = []
        
        # Calculate distances to all cluster centers
        for k in range(NClasses):
            try:
                d = metric(x, ClusterCenters[k])
            except Exception:
                d = 0
            distances.append(d)
        
        # Check for zero distances (Data point coincides with cluster center)
        idx = np.where(np.array(distances) == 0)[0]
        
        if len(idx) == 0:
            # Update membership values using fuzzy formula
            for j in range(NClasses):
                den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(NClasses)])
                MembershipMat[i][j] = float(1 / den)
        else:
            # Point coincides with cluster center: membership = 1 for that cluster
            for j in range(NClasses):
                MembershipMat[i][j] = 0
            MembershipMat[i][idx[0]] = 1
    
    return MembershipMat


def GetClusters(MembershipMat, NPoints):
    """Extract cluster labels from membership matrix.
    
    Assigns each point to the cluster with maximum membership value.
    
    Args:
        MembershipMat (list): membership matrix of shape [NPoints, NClasses]
        NPoints (int): number of Data points
    
    Returns:
        list: cluster labels for each Data point
    """
    ClusterLabels = []
    for i in range(NPoints):
        MaxVal, idx = max((val, idx) for (idx, val) in enumerate(MembershipMat[i]))
        ClusterLabels.append(idx)
    
    return ClusterLabels


def FuzzyCMeansClustering(Data, NPoints, NClasses, centers, FuzzyParameter, MaxIter, metric):
    """Perform Fuzzy C-Means clustering algorithm.
    
    Args:
        Data (array): Dataset of shape [NPoints, NFeatures]
        NPoints (int): number of Data points
        NClasses (int): number of clusters
        centers (list): initial cluster centers
        FuzzyParameter (float): fuzziness parameter (m > 1)
        MaxIter (int): maximum number of iterations
        metric (function): distance metric function
    
    Returns:
        tuple: (ClusterLabels, ClusterCenters, IterationHistory, MembershipMatrix)
            - ClusterLabels (list): final cluster assignment for each point
            - ClusterCenters (list): final cluster centers
            - IterationHistory (list): cluster labels at each iteration
            - MembershipMat (list): final membership matrix
    """
    # Initialize membership matrix
    MembershipMat = InitializeMembershipMatrix(NPoints, NClasses)
    Curr = 0
    Acc = []
    CentTemp = centers
    
    while Curr < MaxIter:
        if Curr == 0:
            ClusterCenters = CentTemp
        else:
            ClusterCenters = CalculateClusterCenters(Data, MembershipMat, NPoints, NClasses, FuzzyParameter)
        
        # Update memberships and get labels
        MembershipMat = UpdateMembershipValue(Data, MembershipMat, NPoints, NClasses, 
                                             FuzzyParameter, ClusterCenters, metric)
        ClusterLabels = GetClusters(MembershipMat, NPoints)
        Acc.append(ClusterLabels)
        Curr += 1
    
    return ClusterLabels, ClusterCenters, Acc, MembershipMat


def FCM(Data, NClasses, centers, FuzzyParameter, MaxIter, metric):
    """Fuzzy C-Means clustering wrapper function.
    
    Performs fuzzy clustering and returns cluster assignments and centers.
    
    Args:
        Data (array): Dataset of shape [NPoints, NFeatures]
        NClasses (int): number of clusters
        Centers (list): initial cluster centers of shape [NClasses, NFeatures]
        FuzzyParameter (float): fuzziness parameter (m > 1, typically 2)
        MaxIter (int): maximum number of iterations
        Metric (function): distance metric function taking (point1, point2) as arguments
    
    Returns:
        tuple: (Clusters, Centers, MembershipMatrix)
            - Clusters (list): list of arrays, each containing indices of points in that cluster
            - Centers (list): final cluster centers
            - MembershipMatrix (list): final fuzzy membership matrix
    """
    NPoints = len(Data)
    
    # Run FCM algorithm
    Labels, Centers, Acc, MembershipMat = FuzzyCMeansClustering(
        Data, NPoints, NClasses, centers, FuzzyParameter, MaxIter, metric
    )
    
    # Convert labels to cluster index lists
    Clusters = []
    labels = np.array(Labels)
    for i in range(NClasses):
        indexes = np.where(labels == i)[0]
        Clusters.append(indexes)
    
    return Clusters, Centers, MembershipMat