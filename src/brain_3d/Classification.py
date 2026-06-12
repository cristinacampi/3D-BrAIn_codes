"""Classification utilities for extracted 3D-BrAIn features."""

import os
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _as_path(path):
    """Return a Path object while accepting strings and Path-like inputs."""
    return Path(path).expanduser()


def _flatten_feature(feature):
    """Convert one feature matrix/vector to a flat 1D numpy array."""
    return np.asarray(feature).reshape(-1)


def _to_dense_array(value):
    """Convert dense or scipy sparse array-like data to a numpy array."""
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray())
    return np.asarray(value)


def _require_sktime_rocket():
    """Import sktime MiniRocket classes only when ROCKET features are requested."""
    CacheDir = Path(tempfile.gettempdir()) / "numba_cache"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(CacheDir))

    try:
        from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
    except ImportError as Exc:
        raise ImportError(
            "MiniRocket support requires sktime. Install it with `pip install sktime` "
            "or add sktime to the active environment."
        ) from Exc

    return MiniRocket, MiniRocketMultivariate


def _as_time_series_panel(X):
    """Convert time-series input to ``(n_samples, n_channels, n_timepoints)``."""
    if isinstance(X, (list, tuple)):
        X = np.stack([_to_dense_array(Item) for Item in X])
    else:
        X = _to_dense_array(X)

    if X.dtype == object and X.ndim == 1:
        X = np.stack([_to_dense_array(Item) for Item in X])

    if X.ndim == 2:
        return X[:, np.newaxis, :]
    if X.ndim == 3:
        return X
    raise ValueError("Time-series input must have shape (samples, timepoints) or (samples, channels, timepoints).")


def tss_score(conf_matrix):
    """Compute the True Skill Statistic from a binary confusion matrix.

    Args:
        conf_matrix (array-like): binary confusion matrix with shape ``(2, 2)``.

    Returns:
        float: true positive rate minus false positive rate.
    """
    conf_matrix = np.asarray(conf_matrix)
    if conf_matrix.shape != (2, 2):
        raise ValueError("tss_score expects a binary 2x2 confusion matrix.")

    Tn, Fp, Fn, Tp = conf_matrix.ravel()
    Sensitivity = Tp / (Tp + Fn) if (Tp + Fn) else 0.0
    FalsePositiveRate = Fp / (Fp + Tn) if (Fp + Tn) else 0.0
    return Sensitivity - FalsePositiveRate


def load_feature_pickles(file_loc, feature_column="Features"):
    """Load extracted-feature pickle files from a directory.

    Each pickle is expected to contain a pandas DataFrame with a feature column
    and one or more metadata/label columns.

    Args:
        file_loc (str or pathlib.Path): directory containing ``.pkl``/``.pickle`` files.
        feature_column (str): name of the DataFrame column containing features.

    Returns:
        pandas.DataFrame: concatenated rows with ``Source_File`` and ``Source_Row`` columns.
    """
    file_loc = _as_path(file_loc)
    Rows = []

    Paths = sorted(file_loc.glob("*.pkl")) + sorted(file_loc.glob("*.pickle"))
    for Path in Paths:
        try:
            Df = pd.read_pickle(Path)
        except Exception as Exc:
            print(f"Error loading {Path.name}: {Exc}")
            continue

        if feature_column not in Df.columns:
            print(f"Skipping {Path.name}: missing column {feature_column!r}")
            continue

        Df = Df.copy()
        Df["Source_File"] = Path.name
        Df["Source_Row"] = np.arange(Df.shape[0])
        Rows.append(Df)

    if not Rows:
        raise ValueError(f"No readable feature pickle files found in {file_loc}.")

    return pd.concat(Rows, ignore_index=True)


def build_classification_dataset(
    data,
    label,
    class_tags,
    pre_label=None,
    pre_tags=None,
    feature_column="Features",
    flatten=True,
):
    """Build ``X`` and ``y`` arrays from extracted features and label tags.

    Args:
        data (pandas.DataFrame): feature table, for example from :func:`load_feature_pickles`.
        label (str): column used as the classification target.
        class_tags (list[list[str]] or dict): class definitions. If a list is used,
            labels are ``0..n-1``. If a dict is used, dict keys become class labels.
        pre_label (str, optional): metadata column used to filter rows before classification.
        pre_tags (list[str], optional): accepted values for ``pre_label``.
        feature_column (str): column containing feature vectors or matrices.
        flatten (bool): if True, flatten each feature entry. Set to False for
            raw time-series matrices passed to MiniRocket.

    Returns:
        tuple: ``(X, y, metadata)`` where ``X`` is a 2D feature matrix, ``y`` is a
        1D label array, and ``metadata`` stores source file/row information.
    """
    if pre_label is not None and pre_tags is not None:
        pre_tags = {str(Tag) for Tag in pre_tags}
        data = data[data[pre_label].astype(str).isin(pre_tags)]

    if isinstance(class_tags, dict):
        ClassItems = list(class_tags.items())
    else:
        ClassItems = list(enumerate(class_tags))

    Features = []
    Labels = []
    Metadata = []

    for Index, Row in data.iterrows():
        RowLabel = str(Row[label])
        MatchedClass = None

        for ClassValue, Tags in ClassItems:
            if RowLabel in {str(Tag) for Tag in Tags}:
                MatchedClass = ClassValue
                break

        if MatchedClass is None:
            continue

        Feature = Row[feature_column]
        Features.append(_flatten_feature(Feature) if flatten else np.asarray(Feature))
        Labels.append(MatchedClass)
        Metadata.append(
            {
                "Index": Index,
                "Source_File": Row.get("Source_File"),
                "Source_Row": Row.get("Source_Row"),
                "Original_Label": Row[label],
            }
        )

    if not Features:
        raise ValueError("No samples matched the requested labels/tags.")

    return np.asarray(Features), np.asarray(Labels), pd.DataFrame(Metadata)


def extract_minirocket_features(
    X_train,
    X_test=None,
    num_kernels=10000,
    multivariate=True,
    channel_group_size=None,
):
    """Fit MiniRocket on train data and transform train/test time series.

    Args:
        X_train (array-like): train signals with shape ``(samples, timepoints)``
            or ``(samples, channels, timepoints)``.
        X_test (array-like, optional): test signals with the same channel/time layout.
        num_kernels (int): number of MiniRocket kernels.
        multivariate (bool): use ``MiniRocketMultivariate`` for multichannel data.
        channel_group_size (int, optional): if set, fit one MiniRocket extractor
            per channel block and concatenate the resulting features. This mirrors
            the chunked strategy used in the original repository scripts.

    Returns:
        dict: transformed features and fitted MiniRocket extractor(s).
    """
    MiniRocket, MiniRocketMultivariate = _require_sktime_rocket()
    X_train = _as_time_series_panel(X_train)
    X_test = None if X_test is None else _as_time_series_panel(X_test)

    if X_test is not None and X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("X_train and X_test must have the same channel/timepoint dimensions.")

    NChannels = X_train.shape[1]
    if channel_group_size is None:
        ChannelSlices = [slice(0, NChannels)]
    else:
        ChannelSlices = [
            slice(Start, min(Start + channel_group_size, NChannels))
            for Start in range(0, NChannels, channel_group_size)
        ]

    TrainFeatures = []
    TestFeatures = []
    Extractors = []

    for ChannelSlice in ChannelSlices:
        XiTrain = X_train[:, ChannelSlice, :]
        XiTest = None if X_test is None else X_test[:, ChannelSlice, :]

        if multivariate or XiTrain.shape[1] > 1:
            Extractor = MiniRocketMultivariate(num_kernels=num_kernels)
            FitTrain = XiTrain
            TransformTrain = XiTrain
            TransformTest = XiTest
        else:
            Extractor = MiniRocket(num_kernels=num_kernels)
            FitTrain = XiTrain[:, 0, :]
            TransformTrain = XiTrain[:, 0, :]
            TransformTest = None if XiTest is None else XiTest[:, 0, :]

        Extractor.fit(FitTrain)
        Extractors.append(Extractor)
        TrainFeatures.append(np.asarray(Extractor.transform(TransformTrain)))
        if TransformTest is not None:
            TestFeatures.append(np.asarray(Extractor.transform(TransformTest)))

    Result = {
        "X_train": np.concatenate(TrainFeatures, axis=1),
        "extractors": Extractors,
        "channel_slices": ChannelSlices,
    }
    if X_test is not None:
        Result["X_test"] = np.concatenate(TestFeatures, axis=1)

    return Result


def minirocket_classifier_cv(
    X,
    y,
    n_splits=25,
    test_size=0.2,
    alphas=None,
    random_state=None,
    num_kernels=10000,
    multivariate=True,
    channel_group_size=None,
    scale_features=False,
    with_mean=False,
):
    """Repeated stratified classification using MiniRocket features and Ridge.

    MiniRocket is fitted separately inside each train/test split, using only the
    training samples. This keeps the test fold out of the feature-extractor fit.

    Args:
        X (array-like): raw time-series data with shape ``(samples, timepoints)``
            or ``(samples, channels, timepoints)``.
        y (array-like): target labels.
        n_splits (int): number of repeated train/test splits.
        test_size (float): fraction of samples used for testing.
        alphas (array-like, optional): alpha grid for ``RidgeClassifierCV``.
        random_state (int, optional): seed used to make splits reproducible.
        num_kernels (int): number of MiniRocket kernels.
        multivariate (bool): use the multivariate MiniRocket transformer.
        channel_group_size (int, optional): split channels into blocks and
            concatenate per-block MiniRocket features.
        scale_features (bool): apply ``StandardScaler`` after MiniRocket.
        with_mean (bool): passed to ``StandardScaler`` when scaling is enabled.

    Returns:
        dict: metrics, confusion matrix, predictions, fitted models, scalers,
        and MiniRocket extractors for each fold.
    """
    X = _as_time_series_panel(X)
    y = np.asarray(y)
    alphas = np.logspace(-3, 3, 10) if alphas is None else alphas
    Classes = np.unique(y)
    Average = "binary" if len(Classes) == 2 and 1 in Classes else "weighted"
    CumulativeConfusionMatrix = np.zeros((len(Classes), len(Classes)), dtype=float)
    Rng = np.random.default_rng(random_state)

    Records = []
    TrainPredictions = []
    TestPredictions = []
    TrainLabels = []
    TestLabels = []
    Models = []
    Scalers = []
    RocketExtractors = []

    for FoldIndex in range(n_splits):
        SplitSeed = None if random_state is None else int(Rng.integers(0, 2**32 - 1))
        XTrain, XTest, YTrain, YTest = train_test_split(
            X,
            y,
            shuffle=True,
            stratify=y,
            test_size=test_size,
            random_state=SplitSeed,
        )

        RocketResult = extract_minirocket_features(
            XTrain,
            XTest,
            num_kernels=num_kernels,
            multivariate=multivariate,
            channel_group_size=channel_group_size,
        )
        XTrainFeatures = RocketResult["X_train"]
        XTestFeatures = RocketResult["X_test"]

        Scaler = None
        if scale_features:
            Scaler = StandardScaler(with_mean=with_mean)
            XTrainFeatures = Scaler.fit_transform(XTrainFeatures)
            XTestFeatures = Scaler.transform(XTestFeatures)

        Clf = RidgeClassifierCV(alphas=alphas)
        Clf.fit(XTrainFeatures, YTrain)

        TrainPred = Clf.predict(XTrainFeatures)
        TestPred = Clf.predict(XTestFeatures)
        TrainConf = confusion_matrix(YTrain, TrainPred, labels=Classes)
        TestConf = confusion_matrix(YTest, TestPred, labels=Classes)
        CumulativeConfusionMatrix += TestConf

        Record = {
            "Fold_Index": FoldIndex,
            "Train_Accuracy": accuracy_score(YTrain, TrainPred),
            "Test_Accuracy": accuracy_score(YTest, TestPred),
            "Train_Precision": precision_score(YTrain, TrainPred, average=Average, zero_division=0),
            "Test_Precision": precision_score(YTest, TestPred, average=Average, zero_division=0),
            "Train_Recall": recall_score(YTrain, TrainPred, average=Average, zero_division=0),
            "Test_Recall": recall_score(YTest, TestPred, average=Average, zero_division=0),
            "Rocket_Features": XTrainFeatures.shape[1],
        }
        if len(Classes) == 2:
            Record["Train_TSS"] = tss_score(TrainConf)
            Record["Test_TSS"] = tss_score(TestConf)

        Records.append(Record)
        TrainPredictions.append(TrainPred)
        TestPredictions.append(TestPred)
        TrainLabels.append(YTrain)
        TestLabels.append(YTest)
        Models.append(Clf)
        Scalers.append(Scaler)
        RocketExtractors.append(RocketResult["extractors"])

    return {
        "metrics": pd.DataFrame(Records),
        "confusion_matrix": CumulativeConfusionMatrix,
        "classes": Classes,
        "train_predictions": TrainPredictions,
        "test_predictions": TestPredictions,
        "train_labels": TrainLabels,
        "test_labels": TestLabels,
        "models": Models,
        "scalers": Scalers,
        "rocket_extractors": RocketExtractors,
    }


def ridge_classifier_cv(
    X,
    y,
    n_splits=25,
    test_size=0.2,
    alphas=None,
    random_state=None,
    with_mean=False,
):
    """Repeated stratified train/test evaluation with ``RidgeClassifierCV``.

    Args:
        X (array-like): feature matrix with shape ``(n_samples, n_features)``.
        y (array-like): target labels.
        n_splits (int): number of repeated train/test splits.
        test_size (float): fraction of samples used for testing.
        alphas (array-like, optional): alpha grid for ridge cross-validation.
        random_state (int, optional): seed used to make the repeated splits reproducible.
        with_mean (bool): passed to :class:`sklearn.preprocessing.StandardScaler`.

    Returns:
        dict: metrics, confusion matrix, predictions, fitted models, and scalers.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    alphas = np.logspace(-3, 3, 10) if alphas is None else alphas
    Classes = np.unique(y)
    Average = "binary" if len(Classes) == 2 and 1 in Classes else "weighted"
    CumulativeConfusionMatrix = np.zeros((len(Classes), len(Classes)), dtype=float)
    Rng = np.random.default_rng(random_state)

    Records = []
    TrainPredictions = []
    TestPredictions = []
    TrainLabels = []
    TestLabels = []
    Models = []
    Scalers = []

    for FoldIndex in range(n_splits):
        SplitSeed = None if random_state is None else int(Rng.integers(0, 2**32 - 1))
        XTrain, XTest, YTrain, YTest = train_test_split(
            X,
            y,
            shuffle=True,
            stratify=y,
            test_size=test_size,
            random_state=SplitSeed,
        )

        Scaler = StandardScaler(with_mean=with_mean)
        XTrain = Scaler.fit_transform(XTrain)
        XTest = Scaler.transform(XTest)

        Clf = RidgeClassifierCV(alphas=alphas)
        Clf.fit(XTrain, YTrain)

        TrainPred = Clf.predict(XTrain)
        TestPred = Clf.predict(XTest)
        TrainConf = confusion_matrix(YTrain, TrainPred, labels=Classes)
        TestConf = confusion_matrix(YTest, TestPred, labels=Classes)
        CumulativeConfusionMatrix += TestConf

        Record = {
            "Fold_Index": FoldIndex,
            "Train_Accuracy": accuracy_score(YTrain, TrainPred),
            "Test_Accuracy": accuracy_score(YTest, TestPred),
            "Train_Precision": precision_score(YTrain, TrainPred, average=Average, zero_division=0),
            "Test_Precision": precision_score(YTest, TestPred, average=Average, zero_division=0),
            "Train_Recall": recall_score(YTrain, TrainPred, average=Average, zero_division=0),
            "Test_Recall": recall_score(YTest, TestPred, average=Average, zero_division=0),
        }

        if len(Classes) == 2:
            Record["Train_TSS"] = tss_score(TrainConf)
            Record["Test_TSS"] = tss_score(TestConf)

        Records.append(Record)
        TrainPredictions.append(TrainPred)
        TestPredictions.append(TestPred)
        TrainLabels.append(YTrain)
        TestLabels.append(YTest)
        Models.append(Clf)
        Scalers.append(Scaler)

    return {
        "metrics": pd.DataFrame(Records),
        "confusion_matrix": CumulativeConfusionMatrix,
        "classes": Classes,
        "train_predictions": TrainPredictions,
        "test_predictions": TestPredictions,
        "train_labels": TrainLabels,
        "test_labels": TestLabels,
        "models": Models,
        "scalers": Scalers,
    }


def plot_classification_results(results, save_loc=None, save_name="classification", show=False):
    """Plot performance boxplots and cumulative confusion matrix.

    Args:
        results (dict): output of :func:`ridge_classifier_cv`.
        save_loc (str or pathlib.Path, optional): directory where plots are saved.
        save_name (str): filename prefix for saved plots.
        show (bool): whether to display plots interactively.
    """
    Metrics = results["metrics"]
    save_loc = None if save_loc is None else _as_path(save_loc)
    if save_loc is not None:
        save_loc.mkdir(parents=True, exist_ok=True)

    MetricColumns = ["Accuracy", "Precision", "Recall"]
    TrainValues = [Metrics[f"Train_{Name}"] for Name in MetricColumns]
    TestValues = [Metrics[f"Test_{Name}"] for Name in MetricColumns]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.boxplot(TrainValues, labels=MetricColumns)
    plt.ylabel("Value")
    plt.title("Train Performance")

    plt.subplot(2, 2, 3)
    plt.boxplot(TestValues, labels=MetricColumns)
    plt.ylabel("Value")
    plt.title("Test Performance")

    if "Train_TSS" in Metrics.columns:
        plt.subplot(2, 2, 2)
        plt.boxplot(Metrics["Train_TSS"], labels=["TSS"])
        plt.ylabel("Value")
        plt.title("Train TSS")

        plt.subplot(2, 2, 4)
        plt.boxplot(Metrics["Test_TSS"], labels=["TSS"])
        plt.ylabel("Value")
        plt.title("Test TSS")

    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc / f"{save_name}_performance.png")
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        results["confusion_matrix"],
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=results["classes"],
        yticklabels=results["classes"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc / f"{save_name}_confusion_matrix.png")
    if show:
        plt.show()
    plt.close()


def classifier(
    file_loc,
    pre_label,
    label,
    pre_tag,
    tag,
    save_loc,
    save_name,
    save_results=False,
    n_splits=25,
    test_size=0.2,
    random_state=None,
    show=False,
    use_minirocket=False,
    feature_column="Features",
    num_kernels=10000,
    multivariate=True,
    channel_group_size=None,
    scale_rocket_features=False,
):
    """Load extracted features and classify them using ``RidgeClassifierCV``.

    This function keeps the call style of the original repository script while
    returning structured results that can be reused downstream.

    Args:
        file_loc (str): directory containing feature DataFrame pickle files.
        pre_label (str): metadata column used for pre-filtering.
        label (str): metadata column used as classification target.
        pre_tag (list[str]): accepted values for ``pre_label``.
        tag (list[list[str]] or dict): class tags used to convert labels to classes.
        save_loc (str): directory where plots/results are saved.
        save_name (str): filename prefix for saved artifacts.
        save_results (bool): whether to save the fold-level results pickle.
        n_splits (int): number of repeated train/test splits.
        test_size (float): fraction of samples used for testing.
        random_state (int, optional): seed for reproducible splits.
        show (bool): whether to display plots interactively.
        use_minirocket (bool): if True, treat ``feature_column`` as raw
            time-series data and extract MiniRocket features inside each split.
        feature_column (str): column containing extracted features or raw signals.
        num_kernels (int): number of MiniRocket kernels when ``use_minirocket`` is True.
        multivariate (bool): use multivariate MiniRocket for multichannel signals.
        channel_group_size (int, optional): split channels into blocks before
            MiniRocket extraction.
        scale_rocket_features (bool): apply ``StandardScaler`` after MiniRocket.

    Returns:
        dict: dataset metadata, metrics, confusion matrix, models, and scalers.
    """
    Data = load_feature_pickles(file_loc, feature_column=feature_column)
    X, Y, Metadata = build_classification_dataset(
        Data,
        label=label,
        class_tags=tag,
        pre_label=pre_label,
        pre_tags=pre_tag,
        feature_column=feature_column,
        flatten=not use_minirocket,
    )
    if use_minirocket:
        Results = minirocket_classifier_cv(
            X,
            Y,
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
            num_kernels=num_kernels,
            multivariate=multivariate,
            channel_group_size=channel_group_size,
            scale_features=scale_rocket_features,
        )
    else:
        Results = ridge_classifier_cv(
            X,
            Y,
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
        )
    Results["metadata"] = Metadata
    Results["X_shape"] = X.shape

    plot_classification_results(Results, save_loc=save_loc, save_name=save_name, show=show)

    if save_results:
        save_loc = _as_path(save_loc)
        save_loc.mkdir(parents=True, exist_ok=True)
        ResultsDf = Results["metrics"].copy()
        ResultsDf["Train_Predictions"] = Results["train_predictions"]
        ResultsDf["Train_Labels"] = Results["train_labels"]
        ResultsDf["Test_Predictions"] = Results["test_predictions"]
        ResultsDf["Test_Labels"] = Results["test_labels"]
        ResultsDf.to_pickle(save_loc / f"{save_name}_results.pkl")

    return Results
