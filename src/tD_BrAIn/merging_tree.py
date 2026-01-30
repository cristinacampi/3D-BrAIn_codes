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
        left (Node): left child node
        right (Node): right child node
        data (str): label or data associated with node
        id (str): unique identifier for the node
    """
    def __init__(self, left=None, right=None, data=None):
        """Initialize a tree node.
        
        Args:
            left (Node, optional): left child node. Defaults to None.
            right (Node, optional): right child node. Defaults to None.
            data (str, optional): node label or data. Defaults to None.
        """
        self.left = left
        self.right = right
        self.data = data
        self.id = str(uuid.uuid4())

    def __repr__(self):
        """String representation of node."""
        return str(self.data)


def BuildGraph(node, G=None, pos=None, x=0, y=0, level_gap=1.5):
    """Build NetworkX directed graph from binary tree structure.
    
    Recursively constructs a graph representation of the tree with hierarchical
    positioning suitable for visualization.
    
    Args:
        node (Node): root node of the tree
        G (nx.DiGraph, optional): graph object to build. Defaults to None.
        pos (dict, optional): node positions for layout. Defaults to None.
        x (float, optional): x-coordinate for current node. Defaults to 0.
        y (float, optional): y-coordinate for current node. Defaults to 0.
        level_gap (float, optional): horizontal spacing between levels. Defaults to 1.5.
    
    Returns:
        tuple: (G, pos) - NetworkX graph and position dictionary
    """
    if G is None:
        G = nx.DiGraph()
        pos = {}

    G.add_node(node.id, label=str(node.data))
    pos[node.id] = (x, y)

    if node.left:
        G.add_edge(node.id, node.left.id)
        BuildGraph(node.left, G, pos, x - level_gap, y - 1, level_gap / 1.5)

    if node.right:
        G.add_edge(node.id, node.right.id)
        BuildGraph(node.right, G, pos, x + level_gap, y - 1, level_gap / 1.5)

    return G, pos


def MergingTree(G, partition):
    """Construct hierarchical merging tree from graph community detection.
    
    Builds a dendrogram-like tree structure by iteratively merging communities
    based on modularity optimization. Uses a modularity-based similarity measure
    to determine which communities should be merged at each step.
    
    Args:
        G (ig.Graph): iGraph graph object
        partition (ig.clustering.VertexPartition): Leiden partition with community membership
    
    Returns:
        tuple: (root, G_tree, pos) where:
            - root (Node): root node of the merging tree
            - G_tree (nx.DiGraph): NetworkX representation of tree
            - pos (dict): node positions for visualization
    """
    # Initialize clusters from partition
    n_community = max(partition.membership) + 1
    clusters = []
    nodi = []
    labels = []
    tree_levels = []
    
    for i in range(n_community):
        idxs = np.where(np.array(partition.membership) == i)[0]
        clusters.append(list(idxs))
        nodi.append(Node(data=str(i)))
        labels.append(str(i))
    
    # Iteratively merge clusters
    while len(clusters) > 2:
        n_community = len(clusters)
        subgraphs = []
        sum_degree = []
        
        # Calculate degree sums for each cluster
        for i in range(n_community):
            subgraphs.append(G.subgraph(clusters[i]))
            sum_degree.append(sum(subgraphs[i].degree()))

        # Calculate inter-community edge weights
        K = np.zeros((n_community, n_community))
        for i in range(n_community):
            for j in range(i + 1, n_community):
                A = G.subgraph(clusters[i])
                B = G.subgraph(clusters[j])
                idxs = np.array(sorted(set(clusters[i]) | set(clusters[j])))
                S = G.subgraph(idxs)
                K[i][j] = len(S.es) - len(A.es) - len(B.es)

        # Calculate modularity-based similarity (gamma)
        gamma = np.zeros((n_community, n_community))
        for i in range(n_community):
            for j in range(i + 1, n_community):
                if sum_degree[i] > 0 and sum_degree[j] > 0:
                    gamma[i][j] = (len(G.es) * K[i][j]) / (sum_degree[i] * sum_degree[j])
        
        M = np.max(gamma)
        tree_levels.append(M)
        
        if M == 0:
            # No more beneficial merges, combine all remaining
            classes = set()
            for i in range(len(clusters) - 1):
                classes = classes | set(clusters[i])
            clusters = [list(sorted(classes)), clusters[-1]]
            
            # Update nodes
            nodi[0] = Node(left=nodi[0], right=nodi[-1], data=labels[0] + labels[-1])
            nodi = [nodi[0], nodi[-1]]
            labels = [labels[0] + labels[-1], labels[-1]]
        else:
            # Find best pair to merge
            idx_del = np.unravel_index(np.argmax(gamma), gamma.shape)
            i_1, i_2 = idx_del[0], idx_del[1]
            
            # Create new merged cluster
            idxs = set(np.arange(n_community)) - {i_1, i_2}
            classes = []
            nodi_new = []
            labels_new = []
            
            # Add merged cluster
            classes.append(list(sorted(set(clusters[i_1]) | set(clusters[i_2]))))
            nodi_new.append(Node(left=nodi[i_1], right=nodi[i_2], 
                                data=labels[i_1] + labels[i_2]))
            labels_new.append(labels[i_1] + labels[i_2])
            
            # Add remaining clusters
            for i in sorted(idxs):
                classes.append(clusters[i])
                nodi_new.append(nodi[i])
                labels_new.append(labels[i])
            
            clusters = classes
            nodi = nodi_new
            labels = labels_new
    
    # Create final root node
    if len(clusters) == 2:
        root = Node(left=nodi[0], right=nodi[1], data=labels[0] + labels[1])
    else:
        root = nodi[0]
    
    # Build NetworkX graph for visualization
    G_tree, pos = BuildGraph(root)
    
    return root, G_tree, pos


def VisualizeTree(G_tree, pos, title="Merging Tree", filename=None):
    """Visualize hierarchical merging tree.
    
    Creates a visual representation of the merging tree using NetworkX and Matplotlib.
    
    Args:
        G_tree (nx.DiGraph): NetworkX directed graph from BuildGraph()
        pos (dict): node positions dictionary from BuildGraph()
        title (str, optional): plot title. Defaults to "Merging Tree".
        filename (str, optional): if provided, saves plot to this file. Defaults to None.
    
    Returns:
        None (displays and optionally saves plot)
    """
    labels = nx.get_node_attributes(G_tree, 'label')
    
    plt.figure(figsize=(12, 8))
    nx.draw(G_tree, pos, labels=labels, with_labels=True, 
            node_size=2000, node_color='skyblue', font_size=10,
            arrows=True, arrowsize=20, edge_color='gray')
    plt.title(title)
    plt.axis('off')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Tree visualization saved to {filename}")
    
    plt.show()


def ExtractClusters(root, depth=None):
    """Extract cluster assignments from merging tree.
    
    Traverses the tree and extracts clusters at a specified depth or leaf level.
    
    Args:
        root (Node): root node of merging tree
        depth (int, optional): depth level to cut tree. If None, uses leaves. Defaults to None.
    
    Returns:
        list: list of clusters (each cluster is a list of node IDs)
    """
    def GetLeafNodes(node, current_depth=0, target_depth=None):
        """Recursively extract nodes at target depth."""
        if node is None:
            return []
        
        if target_depth is not None and current_depth == target_depth:
            return [node.data]
        
        if node.left is None and node.right is None:  # Leaf node
            return [node.data]
        
        left_nodes = GetLeafNodes(node.left, current_depth + 1, target_depth)
        right_nodes = GetLeafNodes(node.right, current_depth + 1, target_depth)
        
        return left_nodes + right_nodes
    
    return GetLeafNodes(root, target_depth=depth)


def TreeHeight(node):
    """Calculate height of merging tree.
    
    Args:
        node (Node): root node of tree
    
    Returns:
        int: height of tree (leaf nodes have height 0)
    """
    if node is None:
        return -1
    
    left_height = TreeHeight(node.left)
    right_height = TreeHeight(node.right)
    
    return 1 + max(left_height, right_height)