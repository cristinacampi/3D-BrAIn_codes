import numpy as np
import torch

from brain_3d.Mamba import MEAMambaClassifier, MambaClassifierCV, MambaClassifierSplit


def test_mea_mamba_forward_shape_and_variable_length():
    model = MEAMambaClassifier(
        n_channels=4, n_classes=3, d_model=8, n_layers=1, d_state=4,
        backend="reference",
    )
    assert model(torch.randn(5, 4, 12)).shape == (5, 3)
    assert model(torch.randn(2, 4, 7)).shape == (2, 3)


def test_group_cv_keeps_recordings_disjoint():
    rng = np.random.default_rng(4)
    # Six recordings, with both classes represented by multiple recordings.
    groups = np.repeat(np.arange(6), 2)
    y = np.repeat([0, 1, 0, 1, 0, 1], 2)
    X = rng.normal(size=(12, 3, 10)).astype(np.float32)
    X[y == 1, 0] += 1.0

    result = MambaClassifierCV(
        X, y, groups=groups, n_splits=1, test_size=0.34, epochs=1,
        batch_size=4, d_model=6, n_layers=1, d_state=3,
        random_state=7,
    )

    split = result["split_indices"][0]
    assert set(groups[split["train"]]).isdisjoint(groups[split["test"]])
    assert result["confusion_matrix"].sum() == len(split["test"])
    assert len(result["metrics"]) == 1


def test_predefined_split_returns_original_labels_and_metrics():
    rng = np.random.default_rng(8)
    X_train = rng.normal(size=(8, 5, 10)).astype(np.float32)
    y_train = np.array(["control", "treated"] * 4)
    X_test = rng.normal(size=(4, 5, 10)).astype(np.float32)
    y_test = np.array(["control", "treated"] * 2)

    result = MambaClassifierSplit(
        X_train, y_train, X_test, y_test=y_test, epochs=1, batch_size=4,
        d_model=6, n_layers=1, d_state=3, random_state=2,
    )

    assert result["test_predictions"].shape == (4,)
    assert result["test_prediction_scores"].shape == (4, 2)
    assert set(result["test_predictions"]).issubset(set(y_train))
    assert "balanced_accuracy" in result["metrics"]
