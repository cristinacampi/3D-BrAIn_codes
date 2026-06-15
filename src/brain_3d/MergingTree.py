"""Hierarchical clustering merging tree construction for community detection."""

import numpy as np
import igraph as ig
import leidenalg as la
import networkx as nx
import uuid
import matplotlib.pyplot as plt


class Node:
    """Binary tree node for representing hierarchical clustering merges.

    Attributes
    ----------
    Left : Node
        Left child node.
    Right : Node
        Right child node.
    Data : str
        Label or data associated with the node.
    ID : str
        Unique node identifier.
    """
    def __init__(self, left=None, Right=None, Data=None):
        """Initialize a tree node.

        Parameters
        ----------
        Left : Node, optional
            Left child node. Defaults to None.
        Right : Node, optional
            Right child node. Defaults to None.
        Data : str, optional
            node label or Data. Defaults to None.
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

    Parameters
    ----------
    node : Node
        root node of the tree
    G : nx.DiGraph, optional
        graph object to build. Defaults to None.
    Pos : dict, optional
        node positions for layout. Defaults to None.
    X : float, optional
        x-coordinate for current node. Defaults to 0.
    Y : float, optional
        y-coordinate for current node. Defaults to 0.
    LevelGap : float, optional
        horizontal spacing between levels. Defaults to 1.5.

    Returns
    -------
    tuple
        (G, Pos) - NetworkX graph and position dictionary
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

    Parameters
    ----------
    G : ig.Graph
        iGraph graph object
    Partition : ig.clustering.VertexPartition
        Leiden partition with community membership

    Returns
    -------
    tuple
        Root node, NetworkX tree representation, and node positions for
        visualization.
    """
    # Initialize Clusters from partition
    NCommunity = max(Partition.membership) + 1
    Clusters = []
    Nodi = []
    Labels = []
    TreeLevels = []

    for I in range(NCommunity):
        Idxs = np.where(np.array(Partition.membership) == I)[0]
        Clusters.append(list(Idxs))
        Nodi.append(Node(Data=str(I)))
        Labels.append(str(I))

    # Iteratively merge Clusters
    while len(Clusters) > 2:
        NCommunity = len(Clusters)
        Subgraphs = []
        SumDegree = []

        # Calculate degree sums for each cluster
        for I in range(NCommunity):
            Subgraphs.append(G.subgraph(Clusters[I]))
            SumDegree.append(sum(Subgraphs[I].degree()))

        # Calculate inter-community edge weights
        K = np.zeros((NCommunity, NCommunity))
        for I in range(NCommunity):
            for J in range(I + 1, NCommunity):
                A = G.subgraph(Clusters[I])
                B = G.subgraph(Clusters[J])
                Idxs = np.array(sorted(set(Clusters[I]) | set(Clusters[J])))
                S = G.subgraph(Idxs)
                K[I][J] = len(S.es) - len(A.es) - len(B.es)

        # Calculate modularity-based similarity (gamma)
        Gamma = np.zeros((NCommunity, NCommunity))
        for I in range(NCommunity):
            for J in range(I + 1, NCommunity):
                if SumDegree[I] > 0 and SumDegree[J] > 0:
                    Gamma[I][J] = (len(G.es) * K[I][J]) / (SumDegree[I] * SumDegree[J])

        M = np.max(Gamma)
        TreeLevels.append(M)

        if M == 0:
            # No more beneficial merges, combine all remaining
            Classes = set()
            for I in range(len(Clusters) - 1):
                Classes = Classes | set(Clusters[I])
            Clusters = [list(sorted(Classes)), Clusters[-1]]

            # Update nodes
            Nodi[0] = Node(left=Nodi[0], Right=Nodi[-1], Data=Labels[0] + Labels[-1])
            Nodi = [Nodi[0], Nodi[-1]]
            Labels = [Labels[0] + Labels[-1], Labels[-1]]
        else:
            # Find best pair to merge
            IdxDel = np.unravel_index(np.argmax(Gamma), Gamma.shape)
            I1, I2 = IdxDel[0], IdxDel[1]

            # Create new merged cluster
            Idxs = set(np.arange(NCommunity)) - {I1, I2}
            Classes = []
            NodiNew = []
            LabelsNew = []

            # Add merged cluster
            Classes.append(list(sorted(set(Clusters[I1]) | set(Clusters[I2]))))
            NodiNew.append(Node(left=Nodi[I1], Right=Nodi[I2],
                                Data=Labels[I1] + Labels[I2]))
            LabelsNew.append(Labels[I1] + Labels[I2])

            # Add remaining Clusters
            for I in sorted(Idxs):
                Classes.append(Clusters[I])
                NodiNew.append(Nodi[I])
                LabelsNew.append(Labels[I])

            Clusters = Classes
            Nodi = NodiNew
            Labels = LabelsNew

    # Create final root node
    if len(Clusters) == 2:
        Root = Node(left=Nodi[0], Right=Nodi[1], Data=Labels[0] + Labels[1])
    else:
        Root = Nodi[0]

    # Build NetworkX graph for visualization
    GTree, Pos = BuildGraph(Root)

    return Root, GTree, Pos


def VisualizeTree(GTree, Pos, Title="Merging Tree", Filename=None):
    """Visualize hierarchical merging tree.

    Creates a visual representation of the merging tree using NetworkX and Matplotlib.

    Parameters
    ----------
    GTree : nx.DiGraph
        NetworkX directed graph from BuildGraph()
    Pos : dict
        node positions dictionary from BuildGraph()
    Title : str, optional
        plot title. Defaults to "Merging Tree".
    Filename : str, optional
        if provided, saves plot to this file. Defaults to None.

    Returns
    -------
    None
        displays and optionally saves plot
    """
    Labels = nx.get_node_attributes(GTree, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(GTree, Pos, labels=Labels, with_labels=True,
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

    Traverses the tree and extracts Clusters at a specified depth or leaf level.

    Parameters
    ----------
    Root : Node
        root node of merging tree
    Depth : int, optional
        depth level to cut tree. If None, uses leaves. Defaults to None.

    Returns
    -------
    list
        list of Clusters (each cluster is a list of node IDs)
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

    Parameters
    ----------
    Node : Node
        root node of tree

    Returns
    -------
    int
        height of tree (leaf nodes have height 0)
    """
    if Node is None:
        return -1

    LeftHeight = TreeHeight(Node.Left)
    RightHeight = TreeHeight(Node.Right)

    return 1 + max(LeftHeight, RightHeight)
