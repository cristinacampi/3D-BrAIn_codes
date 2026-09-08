"""Mamba-style sequence classification for multichannel MEA recordings.

Inputs follow the package convention ``(samples, channels, timepoints)``.  The
module supports the fused ``mamba_ssm`` implementation when it is installed and
a small, portable PyTorch selective state-space implementation for CPU use.
"""

from copy import deepcopy
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _require_torch():
    """Import PyTorch only when the Mamba module is requested."""
    try:
        import torch
        from torch import nn
    except ImportError as Exc:  # pragma: no cover - torch is a package dependency
        raise ImportError("Mamba classification requires PyTorch.") from Exc
    return torch, nn


torch, nn = _require_torch()


def _as_panel(X):
    """Convert MEA time-series input to a dense float32 panel.

    Parameters
    ----------
    X : array-like
        Data with shape ``(samples, timepoints)`` or
        ``(samples, channels, timepoints)``.

    Returns
    -------
    numpy.ndarray
        Panel with shape ``(samples, channels, timepoints)``.
    """
    Panel = np.asarray(X)
    if Panel.ndim == 2:
        Panel = Panel[:, None, :]
    if Panel.ndim != 3:
        raise ValueError("X must have shape (samples, timepoints) or (samples, channels, timepoints).")
    if not np.isfinite(Panel).all():
        raise ValueError("X contains NaN or infinite values.")
    return Panel.astype(np.float32, copy=False)


class SelectiveSSM(nn.Module):
    """Portable selective state-space layer with input-dependent B, C and delta.

    This reference scan prioritizes portability and clarity over speed.  For long
    recordings on NVIDIA GPUs, use ``backend='official'`` in :class:`MambaBlock`.
    """

    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=3, padding=2,
            groups=self.d_inner,
        )
        self.param_proj = nn.Linear(self.d_inner, 1 + 2 * d_state)
        self.log_A = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))[None, :].repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, X):
        """Apply the selective state-space recurrence to one tensor batch."""
        Value, Gate = self.in_proj(X).chunk(2, dim=-1)
        # Causal depth-wise convolution; remove the two padded future positions.
        Value = self.conv(Value.transpose(1, 2))[..., : X.shape[1]].transpose(1, 2)
        Value = torch.nn.functional.silu(Value)
        DeltaRaw, B, C = torch.split(
            self.param_proj(Value), [1, self.d_state, self.d_state], dim=-1
        )
        Delta = torch.nn.functional.softplus(DeltaRaw)
        A = -torch.exp(self.log_A)
        State = Value.new_zeros(Value.shape[0], self.d_inner, self.d_state)
        Outputs = []
        for TimeIndex in range(Value.shape[1]):
            Dt = Delta[:, TimeIndex, :].unsqueeze(-1)
            Decay = torch.exp(Dt * A.unsqueeze(0))
            State = (
                Decay * State
                + Dt * B[:, TimeIndex, :].unsqueeze(1) * Value[:, TimeIndex, :].unsqueeze(-1)
            )
            Y = (State * C[:, TimeIndex, :].unsqueeze(1)).sum(-1)
            Y += self.D * Value[:, TimeIndex, :]
            Outputs.append(Y)
        Y = torch.stack(Outputs, dim=1)
        return self.out_proj(Y * torch.nn.functional.silu(Gate))


class MambaBlock(nn.Module):
    """Pre-normalized residual Mamba block."""

    def __init__(self, d_model, d_state=16, expand=2, backend="reference"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if backend == "reference":
            self.mixer = SelectiveSSM(d_model, d_state=d_state, expand=expand)
        elif backend == "official":
            try:
                from mamba_ssm import Mamba
            except ImportError as Exc:
                raise ImportError(
                    "backend='official' requires mamba-ssm. Install the CUDA-compatible "
                    "package following https://github.com/state-spaces/mamba."
                ) from Exc
            self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
        else:
            raise ValueError("backend must be 'reference' or 'official'.")

    def forward(self, X):
        """Apply the normalized Mamba mixer and residual connection."""
        return X + self.mixer(self.norm(X))


class MEAMambaClassifier(nn.Module):
    """Sequence classifier for MEA windows.

    Parameters
    ----------
    n_channels : int
        Number of electrodes/features at each time point.
    n_classes : int
        Number of target classes.
    d_model : int
        Latent feature dimension.
    n_layers : int
        Number of residual Mamba blocks.
    d_state : int
        State dimension of each selective state-space layer.
    expand : int
        Expansion factor inside each Mamba block.
    dropout : float
        Dropout probability applied before the classification head.
    backend : {"reference", "official"}
        Selective scan implementation.
    """

    def __init__(
        self, n_channels, n_classes, d_model=64, n_layers=2, d_state=16,
        expand=2, dropout=0.1, backend="reference",
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_channels, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, expand=expand, backend=backend)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, X):
        """Return class logits for a batch of multichannel MEA windows."""
        if X.ndim != 3:
            raise ValueError("Expected input with shape (batch, channels, timepoints).")
        X = self.input_projection(X.transpose(1, 2))
        for Block in self.blocks:
            X = Block(X)
        # Mean pooling lets the classifier accept different window lengths.
        return self.head(self.dropout(self.norm(X).mean(dim=1)))


def _split_indices(Y, TestSize, Seed, Groups=None):
    """Create one stratified split, optionally keeping groups disjoint."""
    Indices = np.arange(len(Y))
    if Groups is None:
        TrainIndices, TestIndices = train_test_split(
            Indices, test_size=TestSize, random_state=Seed, stratify=Y,
        )
        return TrainIndices, TestIndices

    Groups = np.asarray(Groups)
    if len(Groups) != len(Y):
        raise ValueError("groups must contain one group ID per sample.")
    Splitter = GroupShuffleSplit(n_splits=32, test_size=TestSize, random_state=Seed)
    Classes = set(np.unique(Y))
    for TrainIndices, TestIndices in Splitter.split(Indices, Y, Groups):
        if set(np.unique(Y[TrainIndices])) == Classes and set(np.unique(Y[TestIndices])) == Classes:
            return TrainIndices, TestIndices
    raise ValueError("Could not create a group-disjoint split containing every class in train and test.")


def _standardize_from_train(X_train, X_test):
    """Standardize panels using per-channel training statistics only."""
    Mean = X_train.mean(axis=(0, 2), keepdims=True)
    Std = X_train.std(axis=(0, 2), keepdims=True)
    Std[Std < 1e-8] = 1.0
    return (X_train - Mean) / Std, (X_test - Mean) / Std, Mean, Std


def _predict(Model, X, BatchSize, Device, ReturnScores=False):
    """Predict labels or probabilities from a fitted PyTorch model."""
    Model.eval()
    Result = []
    with torch.no_grad():
        for Start in range(0, len(X), BatchSize):
            XBatch = torch.from_numpy(X[Start:Start + BatchSize]).to(Device)
            Logits = Model(XBatch)
            Values = torch.softmax(Logits, dim=1) if ReturnScores else Logits.argmax(dim=1)
            Result.append(Values.cpu().numpy())
    return np.concatenate(Result)


def MambaClassifierCV(
    X, y, groups=None, n_splits=5, test_size=0.2, random_state=0,
    epochs=30, batch_size=32, learning_rate=1e-3, weight_decay=1e-4,
    d_model=64, n_layers=2, d_state=16, expand=2, dropout=0.1,
    backend="reference", device=None, verbose=False,
):
    """Train/evaluate Mamba repeatedly, optionally keeping groups disjoint.

    ``groups`` should identify the biological unit (recording, well, organoid or
    subject). Supplying it prevents windows from the same unit appearing in both
    train and test sets.

    Parameters
    ----------
    X : array-like
        MEA windows with shape ``(samples, timepoints)`` or
        ``(samples, channels, timepoints)``.
    y : array-like
        Target label for each sample.
    groups : array-like, optional
        Biological group identifier for each sample.
    n_splits : int
        Number of repeated train/test evaluations.
    test_size : float
        Approximate fraction of samples or groups assigned to testing.
    random_state : int, optional
        Seed used for splitting, initialization, and batch shuffling.
    epochs : int
        Training epochs per split.
    batch_size : int
        Training and inference batch size.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    d_model, n_layers, d_state, expand, dropout
        Parameters passed to :class:`MEAMambaClassifier`.
    backend : {"reference", "official"}
        Selective scan implementation.
    device : str or torch.device, optional
        Execution device. If omitted, CUDA is selected when available.
    verbose : bool
        Print per-epoch training loss.

    Returns
    -------
    dict
        Fold metrics, cumulative confusion matrix, predictions, fitted models,
        split indices, and training normalizers.
    """
    X = _as_panel(X)
    YRaw = np.asarray(y)
    if len(X) != len(YRaw):
        raise ValueError("X and y must contain the same number of samples.")
    Classes, YEncoded = np.unique(YRaw, return_inverse=True)
    if len(Classes) < 2:
        raise ValueError("At least two classes are required.")
    Device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Rng = np.random.default_rng(random_state)
    Records, Models, Predictions, PredictionScores = [], [], [], []
    Labels, SplitIndices, Normalizers = [], [], []
    CumulativeConfusionMatrix = np.zeros((len(Classes), len(Classes)), dtype=int)

    for FoldIndex in range(n_splits):
        SplitSeed = int(Rng.integers(0, 2**31 - 1))
        torch.manual_seed(SplitSeed)
        TrainIndices, TestIndices = _split_indices(YEncoded, test_size, SplitSeed, Groups=groups)
        XTrain, XTest, Mean, Std = _standardize_from_train(X[TrainIndices], X[TestIndices])
        Model = MEAMambaClassifier(
            X.shape[1], len(Classes), d_model=d_model, n_layers=n_layers,
            d_state=d_state, expand=expand, dropout=dropout, backend=backend,
        ).to(Device)
        Optimizer = torch.optim.AdamW(Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        Criterion = nn.CrossEntropyLoss()
        Generator = torch.Generator().manual_seed(SplitSeed)
        Dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(XTrain), torch.from_numpy(YEncoded[TrainIndices]).long(),
        )
        Loader = torch.utils.data.DataLoader(
            Dataset, batch_size=batch_size, shuffle=True, generator=Generator
        )
        Started = time.perf_counter()
        Model.train()
        for EpochIndex in range(epochs):
            Losses = []
            for XBatch, YBatch in Loader:
                XBatch, YBatch = XBatch.to(Device), YBatch.to(Device)
                Optimizer.zero_grad(set_to_none=True)
                Loss = Criterion(Model(XBatch), YBatch)
                Loss.backward()
                Optimizer.step()
                Losses.append(float(Loss.detach().cpu()))
            if verbose:
                print(
                    f"fold={FoldIndex + 1} epoch={EpochIndex + 1} loss={np.mean(Losses):.5f}",
                    flush=True,
                )

        TrainingSeconds = time.perf_counter() - Started
        TrainPredictions = _predict(Model, XTrain, batch_size, Device)
        InferenceStarted = time.perf_counter()
        TestScores = _predict(Model, XTest, batch_size, Device, ReturnScores=True)
        InferenceSeconds = time.perf_counter() - InferenceStarted
        TestPredictions = TestScores.argmax(axis=1)
        CumulativeConfusionMatrix += confusion_matrix(
            YEncoded[TestIndices], TestPredictions, labels=np.arange(len(Classes))
        )
        Average = "binary" if len(Classes) == 2 else "weighted"
        Records.append({
            "Fold_Index": FoldIndex,
            "Train_Accuracy": accuracy_score(YEncoded[TrainIndices], TrainPredictions),
            "Test_Accuracy": accuracy_score(YEncoded[TestIndices], TestPredictions),
            "Train_Precision": precision_score(
                YEncoded[TrainIndices], TrainPredictions, average=Average, zero_division=0
            ),
            "Train_Recall": recall_score(
                YEncoded[TrainIndices], TrainPredictions, average=Average, zero_division=0
            ),
            "Test_Balanced_Accuracy": balanced_accuracy_score(
                YEncoded[TestIndices], TestPredictions
            ),
            "Test_F1": f1_score(
                YEncoded[TestIndices], TestPredictions, average=Average, zero_division=0
            ),
            "Test_Precision": precision_score(
                YEncoded[TestIndices], TestPredictions, average=Average, zero_division=0
            ),
            "Test_Recall": recall_score(
                YEncoded[TestIndices], TestPredictions, average=Average, zero_division=0
            ),
            "Training_Seconds": TrainingSeconds,
            "Inference_Seconds": InferenceSeconds,
        })
        Models.append(deepcopy(Model).cpu())
        Predictions.append(Classes[TestPredictions])
        PredictionScores.append(TestScores)
        Labels.append(YRaw[TestIndices])
        SplitIndices.append({"train": TrainIndices, "test": TestIndices})
        Normalizers.append({"mean": Mean, "std": Std})
        del Model

    return {
        "metrics": pd.DataFrame(Records),
        "confusion_matrix": CumulativeConfusionMatrix,
        "classes": Classes,
        "test_predictions": Predictions,
        "test_prediction_scores": PredictionScores,
        "test_labels": Labels,
        "split_indices": SplitIndices,
        "models": Models,
        "normalizers": Normalizers,
        "backend": backend,
        "device": str(Device),
    }


def MambaClassifierSplit(
    X_train,
    y_train,
    X_test,
    y_test=None,
    epochs=30,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-4,
    d_model=64,
    n_layers=2,
    d_state=16,
    expand=2,
    dropout=0.1,
    backend="reference",
    device=None,
    random_state=0,
    verbose=False,
):
    """Train Mamba on one predefined split and predict its test samples.

    This helper is intended for experiment mains that already create grouped
    cross-validation folds shared by several algorithms. Normalization is fitted
    exclusively on ``X_train``.

    Parameters
    ----------
    X_train, X_test : array-like
        Training and test panels with shape ``(samples, channels, timepoints)``.
    y_train : array-like
        Training labels.
    y_test : array-like, optional
        Test labels used only to calculate returned metrics.
    epochs, batch_size, learning_rate, weight_decay
        Optimization parameters.
    d_model, n_layers, d_state, expand, dropout
        Parameters passed to :class:`MEAMambaClassifier`.
    backend : {"reference", "official"}
        Selective scan implementation.
    device : str or torch.device, optional
        Execution device. If omitted, CUDA is selected when available.
    random_state : int
        Seed used for initialization and batch shuffling.
    verbose : bool
        Print per-epoch training loss.

    Returns
    -------
    dict
        Predictions, probabilities, optional metrics, fitted model,
        normalizer, and execution times.
    """
    XTrain = _as_panel(X_train)
    XTest = _as_panel(X_test)
    if XTrain.shape[1:] != XTest.shape[1:]:
        raise ValueError("X_train and X_test must have matching channel/timepoint dimensions.")

    YTrainRaw = np.asarray(y_train)
    YTestRaw = None if y_test is None else np.asarray(y_test)
    if len(XTrain) != len(YTrainRaw):
        raise ValueError("X_train and y_train must contain the same number of samples.")
    if YTestRaw is not None and len(XTest) != len(YTestRaw):
        raise ValueError("X_test and y_test must contain the same number of samples.")

    Classes = np.unique(YTrainRaw)
    if len(Classes) < 2:
        raise ValueError("At least two training classes are required.")
    ClassToIndex = {ClassValue: Index for Index, ClassValue in enumerate(Classes)}
    YTrain = np.asarray([ClassToIndex[Value] for Value in YTrainRaw], dtype=int)
    if YTestRaw is not None and not set(np.unique(YTestRaw)).issubset(ClassToIndex):
        raise ValueError("y_test contains a class that is absent from y_train.")

    XTrain, XTest, Mean, Std = _standardize_from_train(XTrain, XTest)
    Device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(random_state)
    Model = MEAMambaClassifier(
        XTrain.shape[1], len(Classes), d_model=d_model, n_layers=n_layers,
        d_state=d_state, expand=expand, dropout=dropout, backend=backend,
    ).to(Device)
    Optimizer = torch.optim.AdamW(
        Model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    Criterion = nn.CrossEntropyLoss()
    Generator = torch.Generator().manual_seed(random_state)
    Dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(XTrain), torch.from_numpy(YTrain).long()
    )
    Loader = torch.utils.data.DataLoader(
        Dataset, batch_size=batch_size, shuffle=True, generator=Generator
    )

    Started = time.perf_counter()
    Model.train()
    for EpochIndex in range(epochs):
        Losses = []
        for XBatch, YBatch in Loader:
            XBatch, YBatch = XBatch.to(Device), YBatch.to(Device)
            Optimizer.zero_grad(set_to_none=True)
            Loss = Criterion(Model(XBatch), YBatch)
            Loss.backward()
            Optimizer.step()
            Losses.append(float(Loss.detach().cpu()))
        if verbose:
            print(f"epoch={EpochIndex + 1} loss={np.mean(Losses):.5f}", flush=True)
    TrainingSeconds = time.perf_counter() - Started

    TrainIndices = _predict(Model, XTrain, batch_size, Device)
    InferenceStarted = time.perf_counter()
    TestScores = _predict(Model, XTest, batch_size, Device, ReturnScores=True)
    InferenceSeconds = time.perf_counter() - InferenceStarted
    TestIndices = TestScores.argmax(axis=1)
    TrainPredictions = Classes[TrainIndices]
    TestPredictions = Classes[TestIndices]
    Metrics = None
    if YTestRaw is not None:
        Metrics = _comparison_metrics(YTestRaw, TestPredictions, TestScores)

    return {
        "train_predictions": TrainPredictions,
        "test_predictions": TestPredictions,
        "test_prediction_scores": TestScores,
        "metrics": Metrics,
        "classes": Classes,
        "model": deepcopy(Model).cpu(),
        "normalizer": {"mean": Mean, "std": Std},
        "training_seconds": TrainingSeconds,
        "inference_seconds": InferenceSeconds,
        "device": str(Device),
    }


def _comparison_metrics(YTrue, YPredicted, Scores=None):
    """Compute the common metrics used by every benchmark classifier."""
    Classes = np.unique(YTrue)
    Average = "binary" if len(Classes) == 2 else "weighted"
    MetricOptions = {"average": Average, "zero_division": 0}
    if len(Classes) == 2:
        MetricOptions["pos_label"] = Classes[-1]
    ConfusionMatrix = confusion_matrix(YTrue, YPredicted, labels=Classes)
    Result = {
        "accuracy": accuracy_score(YTrue, YPredicted),
        "balanced_accuracy": balanced_accuracy_score(YTrue, YPredicted),
        "f1": f1_score(YTrue, YPredicted, **MetricOptions),
        "precision": precision_score(YTrue, YPredicted, **MetricOptions),
        "recall": recall_score(YTrue, YPredicted, **MetricOptions),
        "mcc": matthews_corrcoef(YTrue, YPredicted),
        "specificity": np.nan,
        "tss": np.nan,
        "roc_auc": np.nan,
    }
    if len(Classes) == 2 and ConfusionMatrix.shape == (2, 2):
        Tn, Fp, Fn, Tp = ConfusionMatrix.ravel()
        Specificity = Tn / (Tn + Fp) if Tn + Fp else 0.0
        Sensitivity = Tp / (Tp + Fn) if Tp + Fn else 0.0
        Result.update(specificity=Specificity, tss=Sensitivity + Specificity - 1.0)
    if Scores is not None:
        try:
            if len(Classes) == 2:
                Score = Scores[:, 1] if np.ndim(Scores) == 2 else Scores
                Result["roc_auc"] = roc_auc_score(YTrue, Score)
            else:
                Result["roc_auc"] = roc_auc_score(
                    YTrue, Scores, multi_class="ovr", average="weighted"
                )
        except ValueError:
            pass
    return Result


def _model_scores(Model, X):
    """Return probability or decision scores when exposed by a classifier."""
    if hasattr(Model, "predict_proba"):
        return Model.predict_proba(X)
    if hasattr(Model, "decision_function"):
        return Model.decision_function(X)
    return None


def _classical_models(random_state=0, pca_variance=0.90, n_estimators=300):
    """Build fresh classical classifier instances for one benchmark fold."""
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "ridge_pca90": make_pipeline(
            StandardScaler(), PCA(n_components=pca_variance, svd_solver="full"),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        ),
        "logistic_balanced_pca90": make_pipeline(
            StandardScaler(), PCA(n_components=pca_variance, svd_solver="full"),
            LogisticRegression(class_weight="balanced", max_iter=5000),
        ),
        "linear_svm_balanced_pca90": make_pipeline(
            StandardScaler(), PCA(n_components=pca_variance, svd_solver="full"),
            SVC(kernel="linear", class_weight="balanced"),
        ),
        "rbf_svm_balanced_pca90": make_pipeline(
            StandardScaler(), PCA(n_components=pca_variance, svd_solver="full"),
            SVC(kernel="rbf", class_weight="balanced"),
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced", max_features="sqrt",
            random_state=random_state, n_jobs=-1,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=n_estimators, class_weight="balanced", max_features="sqrt",
            random_state=random_state, n_jobs=-1,
        ),
    }


def CompareMambaToClassical(
    X, y, groups=None, pca_variance=0.90, n_estimators=300, **mamba_kwargs,
):
    """Compare Mamba with repository classifiers on identical grouped folds.

    Parameters
    ----------
    X : array-like
        Raw MEA windows. Classical models receive flattened copies of the same
        windows used by Mamba.
    y : array-like
        Target labels.
    groups : array-like, optional
        Biological group identifier for leakage-safe splitting.
    pca_variance : float
        Explained variance retained by PCA-based classical models.
    n_estimators : int
        Number of trees used by Random Forest and Extra Trees.
    **mamba_kwargs
        Additional arguments passed to :func:`MambaClassifierCV`.

    Returns
    -------
    dict
        Fold metrics, ranked summary, sample predictions, cumulative confusion
        matrices, class labels, and the complete Mamba result.
    """
    X = _as_panel(X)
    Y = np.asarray(y)
    MambaResult = MambaClassifierCV(X, Y, groups=groups, **mamba_kwargs)
    Records, Predictions = [], []
    ConfusionMatrices = {"mamba": MambaResult["confusion_matrix"]}
    RandomState = mamba_kwargs.get("random_state", 0)
    for FoldIndex, Split in enumerate(MambaResult["split_indices"]):
        TrainIndices, TestIndices = Split["train"], Split["test"]
        XTrain = X[TrainIndices].reshape(len(TrainIndices), -1)
        XTest = X[TestIndices].reshape(len(TestIndices), -1)
        Models = _classical_models(RandomState + FoldIndex, pca_variance, n_estimators)
        for Name, Model in Models.items():
            Started = time.perf_counter()
            Model.fit(XTrain, Y[TrainIndices])
            TrainingSeconds = time.perf_counter() - Started
            Started = time.perf_counter()
            Predicted = Model.predict(XTest)
            InferenceSeconds = time.perf_counter() - Started
            Metrics = _comparison_metrics(
                Y[TestIndices], Predicted, _model_scores(Model, XTest)
            )
            Records.append({"fold_index": FoldIndex, "model": Name, **Metrics,
                            "training_seconds": TrainingSeconds,
                            "inference_seconds": InferenceSeconds})
            Predictions.extend(
                {"fold_index": FoldIndex, "model": Name, "sample_index": int(Index),
                 "true_label": TrueLabel, "predicted_label": PredictedLabel}
                for Index, TrueLabel, PredictedLabel
                in zip(TestIndices, Y[TestIndices], Predicted)
            )
            ConfusionMatrices.setdefault(
                Name, np.zeros_like(MambaResult["confusion_matrix"])
            )
            ConfusionMatrices[Name] += confusion_matrix(
                Y[TestIndices], Predicted, labels=MambaResult["classes"]
            )

        MambaPredictions = MambaResult["test_predictions"][FoldIndex]
        MambaMetrics = _comparison_metrics(
            Y[TestIndices], MambaPredictions,
            MambaResult["test_prediction_scores"][FoldIndex],
        )
        MambaRow = MambaResult["metrics"].iloc[FoldIndex]
        Records.append({
            "fold_index": FoldIndex,
            "model": "mamba",
            **MambaMetrics,
            "training_seconds": MambaRow["Training_Seconds"],
            "inference_seconds": MambaRow["Inference_Seconds"],
        })
        Predictions.extend(
            {"fold_index": FoldIndex, "model": "mamba", "sample_index": int(Index),
             "true_label": TrueLabel, "predicted_label": PredictedLabel}
            for Index, TrueLabel, PredictedLabel
            in zip(TestIndices, Y[TestIndices], MambaPredictions)
        )

    Comparison = pd.DataFrame(Records)
    Summary = Comparison.groupby("model", sort=False).agg(
        folds=("fold_index", "count"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
        mcc_mean=("mcc", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
        tss_mean=("tss", "mean"),
        training_seconds_mean=("training_seconds", "mean"),
        inference_seconds_mean=("inference_seconds", "mean"),
    ).reset_index()
    Summary = Summary.sort_values(
        ["balanced_accuracy_mean", "f1_mean"], ascending=False
    ).reset_index(drop=True)
    Summary.insert(0, "rank", np.arange(1, len(Summary) + 1))
    return {
        "metrics": Comparison,
        "summary": Summary,
        "predictions": pd.DataFrame(Predictions),
        "confusion_matrices": ConfusionMatrices,
        "classes": MambaResult["classes"],
        "mamba": MambaResult,
    }


def CompareMambaToRidge(X, y, groups=None, **mamba_kwargs):
    """Compare Mamba and Ridge while preserving the original helper API.

    Parameters
    ----------
    X, y, groups
        See :func:`CompareMambaToClassical`.
    **mamba_kwargs
        Additional arguments passed to :func:`MambaClassifierCV`.

    Returns
    -------
    dict
        Comparison output filtered to Mamba and Ridge results.
    """
    Result = CompareMambaToClassical(X, y, groups=groups, **mamba_kwargs)
    Keep = Result["metrics"]["model"].isin(["mamba", "ridge_pca90"])
    Result["metrics"] = Result["metrics"][Keep].reset_index(drop=True)
    SummaryKeep = Result["summary"]["model"].isin(["mamba", "ridge_pca90"])
    Result["summary"] = Result["summary"][SummaryKeep].reset_index(drop=True)
    return Result


# Backward-compatible aliases for releases that exposed snake_case names.
mamba_classifier_cv = MambaClassifierCV
mamba_classifier_split = MambaClassifierSplit
compare_mamba_to_classical = CompareMambaToClassical
compare_mamba_to_ridge = CompareMambaToRidge
