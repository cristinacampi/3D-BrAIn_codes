"""Hierarchical clustering merging tree construction for community detection."""

import numpy as np
import igraph as ig
import leidenalg as la
import networkx as nx
import uuid
import matplotlib.pyplot as plt


class Node:
    """Binary tree node for representing hierarchical clustering merges.
    
    Attributes:
        Left(Node): Left child node
        Right (Node): Right child node
        Data (str): Label or data associated with node
        ID (str): unique identifier for the node
    """
    def __init__(self, left=None, Right=None, Data=None):
        """Initialize a tree node.
        
        Args:
            Left(Node, optional): Left child node. Defaults to None.
            Right (Node, optional): Right child node. Defaults to None.
            Data (str, optional): node label or Data. Defaults to None.
        """
        self.Left= left
        self.Right = Right
        self.Data = Data
        self.ID = str(uuid.uuid4())

    def __repr__(self):
        """String representation of node."""
        return str(self.Data)


def BuildGraph(node, G=None, Pos=None, X=0, Y=0, LevelGap=1.5):
    """Build NetworkX directed graph from binary tree structure.
    
    Recursively constructs a graph representation of the tree with hierarchical
    positioning suitable for visualization.
    
    Args:
        node (Node): root node of the tree
        G (nx.DiGraph, optional): graph object to build. Defaults to None.
        Pos (dict, optional): node positions for layout. Defaults to None.
        X (float, optional): x-coordinate for current node. Defaults to 0.
        Y (float, optional): y-coordinate for current node. Defaults to 0.
        LevelGap (float, optional): horizontal spacing between levels. Defaults to 1.5.
    
    Returns:
        tuple: (G, Pos) - NetworkX graph and position dictionary
    """
    if G is None:
        G = nx.DiGraph()
        Pos = {}

    G.add_node(node.ID, label=str(node.Data))
    Pos[node.ID] = (X, Y)

    if node.Left:
        G.add_edge(node.ID, node.Left.ID)
        BuildGraph(node.Left, G, Pos, X - LevelGap, Y - 1, LevelGap / 1.5)

    if node.Right:
        G.add_edge(node.ID, node.Right.ID)
        BuildGraph(node.Right, G, Pos, X + LevelGap, Y - 1, LevelGap / 1.5)

    return G, Pos


def MergingTree(G, Partition):
    """Construct hierarchical merging tree from graph community detection.
    
    Builds a dendrogram-like tree structure by iteratively merging communities
    based on modularity optimization. Uses a modularity-based similarity measure
    to determine which communities should be merged at each step.
    
    Args:
        G (ig.Graph): iGraph graph object
        Partition (ig.clustering.VertexPartition): Leiden partition with community membership
    
    Returns:
        tuple: (Root, GTree, Pos) where:
            - root (Node): root node of the merging tree
            - GTree (nx.DiGraph): NetworkX representation of tree
            - Pos (dict): node positions for visualization
    """
    # Initialize clusters from partition
    NCommunity = max(Partition.membership) + 1
    Clusters = []
    Nodi = []
    Labels = []
    TreeLevels = []
    
    for i in range(NCommunity):
        idxs = np.where(np.array(Partition.membership) == i)[0]
        Clusters.append(list(idxs))
        Nodi.append(Node(Data=str(i)))
        Labels.append(str(i))
    
    # Iteratively merge clusters
    while len(Clusters) > 2:
        NCommunity = len(Clusters)
        Subgraphs = []
        SumDegree = []
        
        # Calculate degree sums for each cluster
        for i in range(NCommunity):
            Subgraphs.append(G.subgraph(Clusters[i]))
            SumDegree.append(sum(Subgraphs[i].degree()))

        # Calculate inter-community edge weights
        K = np.zeros((NCommunity, NCommunity))
        for i in range(NCommunity):
            for j in range(i + 1, NCommunity):
                A = G.subgraph(Clusters[i])
                B = G.subgraph(Clusters[j])
                idxs = np.array(sorted(set(Clusters[i]) | set(Clusters[j])))
                S = G.subgraph(idxs)
                K[i][j] = len(S.es) - len(A.es) - len(B.es)

        # Calculate modularity-based similarity (gamma)
        gamma = np.zeros((NCommunity, NCommunity))
        for i in range(NCommunity):
            for j in range(i + 1, NCommunity):
                if SumDegree[i] > 0 and SumDegree[j] > 0:
                    gamma[i][j] = (len(G.es) * K[i][j]) / (SumDegree[i] * SumDegree[j])
        
        M = np.max(gamma)
        TreeLevels.append(M)
        
        if M == 0:
            # No more beneficial merges, combine all remaining
            Classes = set()
            for i in range(len(Clusters) - 1):
                Classes = Classes | set(Clusters[i])
            Clusters = [list(sorted(Classes)), Clusters[-1]]
            
            # Update nodes
            Nodi[0] = Node(left=Nodi[0], Right=Nodi[-1], Data=Labels[0] + Labels[-1])
            Nodi = [Nodi[0], Nodi[-1]]
            Labels = [Labels[0] + Labels[-1], Labels[-1]]
        else:
            # Find best pair to merge
            idxDel = np.unravel_index(np.argmax(gamma), gamma.shape)
            i1, i2 = idxDel[0], idxDel[1]
            
            # Create new merged cluster
            idxs = set(np.arange(NCommunity)) - {i1, i2}
            Classes = []
            NodiNew = []
            LabelsNew = []
            
            # Add merged cluster
            Classes.append(list(sorted(set(clusters[i1]) | set(clusters[i2]))))
            NodiNew.append(Node(left=Nodi[i1], Right=Nodi[i2], 
                                Data=Labels[i1] + Labels[i2]))
            LabelsNew.append(Labels[i1] + Labels[i2])
            
            # Add remaining clusters
            for i in sorted(idxs):
                Classes.append(clusters[i])
                NodiNew.append(Nodi[i])
                LabelsNew.append(Labels[i])
            
            Clusters = Classes
            Nodi = NodiNew
            Labels = LabelsNew
    
    # Create final root node
    if len(Clusters) == 2:
        root = Node(left=Nodi[0], Right=Nodi[1], Data=Labels[0] + Labels[1])
    else:
        root = Nodi[0]
    
    # Build NetworkX graph for visualization
    GTree, Pos = BuildGraph(root)
    
    return root, GTree, Pos


def VisualizeTree(GTree, Pos, Title="Merging Tree", Filename=None):
    """Visualize hierarchical merging tree.
    
    Creates a visual representation of the merging tree using NetworkX and Matplotlib.
    
    Args:
        GTree (nx.DiGraph): NetworkX directed graph from BuildGraph()
        Pos (dict): node positions dictionary from BuildGraph()
        Title (str, optional): plot title. Defaults to "Merging Tree".
        Filename (str, optional): if provided, saves plot to this file. Defaults to None.
    
    Returns:
        None (displays and optionally saves plot)
    """
    labels = nx.get_node_attributes(GTree, 'label')
    
    plt.figure(figsize=(12, 8))
    nx.draw(GTree, Pos, labels=labels, with_labels=True, 
            node_size=2000, node_color='skyblue', font_size=10,
            arrows=True, arrowsize=20, edge_color='gray')
    plt.title(Title)
    plt.axis('off')
    
    if Filename:
        plt.savefig(Filename, dpi=300, bbox_inches='tight')
        print(f"Tree visualization saved to {Filename}")
    
    plt.show()


def ExtractClusters(Root, Depth=None):
    """Extract cluster assignments from merging tree.
    
    Traverses the tree and extracts clusters at a specified depth or leaf level.
    
    Args:
        Root (Node): root node of merging tree
        Depth (int, optional): depth level to cut tree. If None, uses leaves. Defaults to None.
    
    Returns:
        list: list of clusters (each cluster is a list of node IDs)
    """
    def GetLeafNodes(Node, CurrentDepth=0, TargetDepth=None):
        """Recursively extract nodes at target depth."""
        if Node is None:
            return []
        
        if TargetDepth is not None and CurrentDepth == TargetDepth:
            return [Node.Data]
        
        if Node.Left is None and Node.Right is None:  # Leaf node
            return [Node.Data]
        
        LeftNodes = GetLeafNodes(Node.Left, CurrentDepth + 1, TargetDepth)
        RightNodes = GetLeafNodes(Node.Right, CurrentDepth + 1, TargetDepth)
        
        return LeftNodes + RightNodes
    
    return GetLeafNodes(Root, TargetDepth=Depth)


def TreeHeight(Node):
    """Calculate height of merging tree.
    
    Args:
        Node (Node): root node of tree
    
    Returns:
        int: height of tree (leaf nodes have height 0)
    """
    if Node is None:
        return -1
    
    LeftHeight = TreeHeight(Node.Left)
    RightHeight = TreeHeight(Node.Right)
    
    return 1 + max(LeftHeight, RightHeight)