"""Fuzzy C-Means clustering implementation."""

import numpy as np
import random
import math


def InitializeMembershipMatrix(NPoints, NClasses):
    """Initialize fuzzy membership matrix with random values.
    
    Creates a matrix where each row represents a Data point and each column 
    represents membership to a cluster. Values are normalized to sum to 1.
    
    Args:
        NPoints (int): number of Data pointsCenters
        NClasses (int): number of clusters
    
    Returns:
        list: membership matrix of shape [NPoints, NClasses] with random normalized values
    """
    MembershipMat = []
    for I in range(NPoints):
        # Generate random membership values
        RandomNumList = [random.random() for _ in range(NClasses)]
        Summation = sum(RandomNumList)
        
        # Normalize to sum to 1
        TempList = [X / Summation for X in RandomNumList]
        
        # Set max value to 1, others to 0 (hard initialization)
        Flag = TempList.index(max(TempList))
        for J in range(len(TempList)):
            TempList[J] = 1 if J == Flag else 0
        
        MembershipMat.append(TempList)
    
    return MembershipMat


def CalculateClusterCenters(Data, MembershipMat, NPoints, NClasses, FuzzyParameter):
    """Calculate cluster Centers based on membership matrix.
    
    Args:
        Data (array): Dataset of shape [NPoints, NFeatures]
        MembershipMat (list): membership matrix of shape [NPoints, NClasses]
        NPoints (int): number of Data points
        NClasses (int): number of clusters
        FuzzyParameter (float): fuzziness parameter (m > 1, typically 2)
    
    Returns:
        list: cluster Centers of shape [NClasses, NFeatures]
    """
    ClusterMemVal = list(zip(*MembershipMat))
    ClusterCenters = []
    
    for J in range(NClasses):
        X = list(ClusterMemVal[J])
        XRaised = [P ** FuzzyParameter for P in X]
        Denominator = sum(XRaised)
        
        # Calculate weighted sum of Data points
        TempNum = []
        for I in range(NPoints):
            DataPoint = list(Data[I])
            Prod = [XRaised[I] * Val for Val in DataPoint]
            TempNum.append(Prod)
        
        Numerator = list(map(sum, list(zip(*TempNum))))
        Center = [Z / Denominator for Z in Numerator]
        ClusterCenters.append(Center)
    
    return ClusterCenters


def UpdateMembershipValue(Data, MembershipMat, NPoints, NClasses, FuzzyParameter, ClusterCenters, Metric):
    """Update membership matrix based on distances to cluster Metric.
    
    Args:
        Data (array): Dataset of shape [NPoints, NFeatures]
        MembershipMat (list): current membership matrix
        NPoints (int): number of Data points
        NClasses (int): number of clusters
        FuzzyParameter (float): fuzziness parameter
        ClusterCenters (list): cluster centers
        Metric (function): distance metric function
    
    Returns:
        list: updated membership matrix
    """
    P = float(2 / (FuzzyParameter - 1))
    
    for I in range(NPoints):
        X = list(Data[I])
        Distances = []
        
        # Calculate distances to all cluster centers
        for K in range(NClasses):
            try:
                D = Metric(X, ClusterCenters[K])
            except Exception:
                D = 0
            Distances.append(D)
        
        # Check for zero distances (Data point coincides with cluster center)
        Idx = np.where(np.array(Distances) == 0)[0]
        
        if len(Idx) == 0:
            # Update membership values using fuzzy formula
            for J in range(NClasses):
                Den = sum([math.pow(float(Distances[J] / Distances[C]), P) for C in range(NClasses)])
                MembershipMat[I][J] = float(1 / Den)
        else:
            # Point coincides with cluster center: membership = 1 for that cluster
            for J in range(NClasses):
                MembershipMat[I][J] = 0
            MembershipMat[I][Idx[0]] = 1
    
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
    for I in range(NPoints):
        MaxVal, Idx = max((Val, Idx) for (Idx, Val) in enumerate(MembershipMat[I]))
        ClusterLabels.append(Idx)
    
    return ClusterLabels


def FuzzyCMeansClustering(Data, NPoints, NClasses, Centers, FuzzyParameter, MaxIter, Metric):
    """Perform Fuzzy C-Means clustering algorithm.
    
    Args:
        Data (array): Dataset of shape [NPoints, NFeatures]
        NPoints (int): number of Data points
        NClasses (int): number of clusters
        Centers (list): initial cluster centers
        FuzzyParameter (float): fuzziness parameter (m > 1)
        MaxIter (int): maximum number of iterations
        Metric (function): distance metric function
    
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
    CentTemp = Centers
    
    while Curr < MaxIter:
        if Curr == 0:
            ClusterCenters = CentTemp
        else:
            ClusterCenters = CalculateClusterCenters(Data, MembershipMat, NPoints, NClasses, FuzzyParameter)
        
        # Update memberships and get labels
        MembershipMat = UpdateMembershipValue(Data, MembershipMat, NPoints, NClasses, 
                                             FuzzyParameter, ClusterCenters, Metric)
        ClusterLabels = GetClusters(MembershipMat, NPoints)
        Acc.append(ClusterLabels)
        Curr += 1
    
    return ClusterLabels, ClusterCenters, Acc, MembershipMat


def FCM(Data, NClasses, Centers, FuzzyParameter, MaxIter, Metric):
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
        Data, NPoints, NClasses, Centers, FuzzyParameter, MaxIter, Metric
    )
    
    # Convert labels to cluster index lists
    Clusters = []
    LabelsArray = np.array(Labels)
    for I in range(NClasses):
        Indexes = np.where(LabelsArray == I)[0]
        Clusters.append(Indexes)
    
    return Clusters, Centers, MembershipMat
