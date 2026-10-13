import numpy as np
import pandas as pd

from brain_3d import Connectivity


def test_edge_node_strength_matches_displayed_absolute_edges():
    edges = pd.DataFrame(
        {
            "source": [1, 1, 2],
            "target": [2, 3, 3],
            "weight": [0.8, -0.4, 0.2],
        }
    )

    strength = Connectivity.EdgeNodeStrength(edges, Absolute=True).set_index("channel")["strength"]

    assert np.isclose(strength.loc[1], 1.2)
    assert np.isclose(strength.loc[2], 1.0)
    assert np.isclose(strength.loc[3], 0.6)


def test_select_strong_edges_uses_absolute_magnitude():
    edges = pd.DataFrame(
        {
            "source": [1, 1, 2],
            "target": [2, 3, 3],
            "weight": [0.3, -0.9, 0.6],
        }
    )

    selected = Connectivity.SelectStrongEdges(edges, TopN=2, Absolute=True)

    assert selected["weight"].tolist() == [-0.9, 0.6]
