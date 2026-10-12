"""
Codes with functions related to Spikes sorting
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from . import Stratification
from . import BrwFunctions as brw_f
import time
from scipy.signal import find_peaks, butter, filtfilt
from statistics import median
from scipy.stats import pearsonr
import igraph as ig
from igraph import Graph, plot
import leidenalg as la
from . import MergingTree as merge
import matplotlib.patches as mpatches
from neo import SpikeTrain
import quantities as pq
from elephant.spike_train_correlation import cross_correlation_histogram
from concurrent.futures import ProcessPoolExecutor
import psutil
import os
import gc
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpikePolarityFilterConfig:
    """Configure detector-label and waveform-polarity event filtering.

    Attributes
    ----------
    polarity : str
        Event polarity to retain: ``"negative"``, ``"positive"``, or ``"none"``.
    negative_spike_label : int
        Detector label assigned to negative events.
    positive_spike_label : int
        Detector label assigned to positive events.
    use_detector_label_filter : bool
        Whether to filter events using detector labels when labels are available.
    use_waveform_polarity_filter : bool
        Whether to require the requested polarity to dominate the waveform.
    polarity_dominance_ratio : float
        Minimum ratio between the requested and opposite waveform extrema.
    require_extremum_near_center : bool
        Whether the dominant extremum must be close to the waveform centre.
    center_window_samples : int
        Maximum distance in samples between the extremum and waveform centre.
    """

    polarity: str = "negative"
    negative_spike_label: int = 0
    positive_spike_label: int = 1
    use_detector_label_filter: bool = True
    use_waveform_polarity_filter: bool = True
    polarity_dominance_ratio: float = 1.10
    require_extremum_near_center: bool = True
    center_window_samples: int = 10


@dataclass(frozen=True)
class NegativeSpikeFilterConfig:
    """Configure the legacy negative-waveform filtering interface.

    Attributes use the same meanings as :class:`SpikePolarityFilterConfig`,
    with polarity fixed to ``"negative"``. The class is retained for backward
    compatibility with existing scripts and saved analysis configurations.
    """

    negative_spike_label: int = 0
    use_detector_label_filter: bool = True
    use_waveform_polarity_filter: bool = True
    polarity_dominance_ratio: float = 1.10
    require_extremum_near_center: bool = True
    center_window_samples: int = 10


def WaveformPolarityMask(Spikes, Labels=None, Config=None):
    """Select events using detector labels and waveform polarity.

    Parameters
    ----------
    Spikes : array-like
        Two-dimensional waveform array with shape ``(events, samples)``.
    Labels : array-like, optional
        Detector labels for the events. Missing or entirely non-finite labels
        leave detector-label filtering disabled for that recording.
    Config : SpikePolarityFilterConfig, optional
        Filtering configuration. Defaults to negative-polarity filtering.

    Returns
    -------
    numpy.ndarray
        Boolean mask with one value per input event.

    Raises
    ------
    ValueError
        If the waveform shape, label count, polarity, or ratio is invalid.
    """
    Spikes = np.asarray(Spikes)
    if Spikes.ndim != 2:
        raise ValueError("Spikes must be a 2-D array of shape (events, samples).")
    Config = Config or SpikePolarityFilterConfig()
    Polarity = str(Config.polarity).strip().lower()
    if Polarity not in {"negative", "positive", "none"}:
        raise ValueError("Config.polarity must be 'negative', 'positive', or 'none'.")
    if Config.polarity_dominance_ratio < 0:
        raise ValueError("Config.polarity_dominance_ratio must be non-negative.")
    if Labels is None:
        Labels = np.full(Spikes.shape[0], np.nan)
    Labels = np.asarray(Labels).reshape(-1)
    if len(Labels) != len(Spikes):
        raise ValueError("Spikes and Labels must contain the same number of events.")
    Keep = np.ones(len(Spikes), dtype=bool)
    if Config.use_detector_label_filter and Polarity != "none" and np.isfinite(Labels).any():
        WantedLabel = Config.negative_spike_label if Polarity == "negative" else Config.positive_spike_label
        Keep &= Labels == WantedLabel
    if Config.use_waveform_polarity_filter and Polarity != "none" and len(Spikes):
        WaveformMin = np.min(Spikes, axis=1)
        WaveformMax = np.max(Spikes, axis=1)
        Center = (Spikes.shape[1] - 1) / 2
        if Polarity == "negative":
            ExtremumIndex = np.argmin(Spikes, axis=1)
            WaveformKeep = np.abs(WaveformMin) >= Config.polarity_dominance_ratio * np.abs(WaveformMax)
        else:
            ExtremumIndex = np.argmax(Spikes, axis=1)
            WaveformKeep = np.abs(WaveformMax) >= Config.polarity_dominance_ratio * np.abs(WaveformMin)
        if Config.require_extremum_near_center:
            WaveformKeep &= np.abs(ExtremumIndex - Center) <= Config.center_window_samples
        Keep &= WaveformKeep
    return Keep


def NegativeWaveformMask(Spikes, Labels=None, Config=None):
    """Select plausible negative spike waveforms.

    Parameters
    ----------
    Spikes : array-like
        Two-dimensional waveform array with shape ``(events, samples)``.
    Labels : array-like, optional
        Detector labels associated with ``Spikes``.
    Config : NegativeSpikeFilterConfig, optional
        Legacy negative-polarity filtering configuration.

    Returns
    -------
    numpy.ndarray
        Boolean mask selecting negative events.
    """
    LegacyConfig = Config or NegativeSpikeFilterConfig()
    GenericConfig = SpikePolarityFilterConfig(
        polarity="negative",
        negative_spike_label=LegacyConfig.negative_spike_label,
        use_detector_label_filter=LegacyConfig.use_detector_label_filter,
        use_waveform_polarity_filter=LegacyConfig.use_waveform_polarity_filter,
        polarity_dominance_ratio=LegacyConfig.polarity_dominance_ratio,
        require_extremum_near_center=LegacyConfig.require_extremum_near_center,
        center_window_samples=LegacyConfig.center_window_samples,
    )
    return WaveformPolarityMask(Spikes, Labels, GenericConfig)


def ChannelFromPath(PathValue):
    """Extract a numeric channel identifier from a processed-array path.

    Parameters
    ----------
    PathValue : path-like
        Path whose stem ends with a channel token such as ``ch42``.

    Returns
    -------
    int or None
        Parsed channel number, or ``None`` when the suffix is not numeric.
    """
    Stem = Path(PathValue).stem
    try:
        return int(Stem.split("ch")[-1])
    except ValueError:
        return None


def LoadSpikeChannel(RecordingDirectory, Channel):
    """Load event arrays for one processed channel.

    Parameters
    ----------
    RecordingDirectory : path-like
        Directory containing per-channel NumPy arrays.
    Channel : int
        Channel identifier used in the array filenames.

    Returns
    -------
    tuple
        ``(Spikes, Frames, Labels)`` arrays. Missing frame or label files are
        represented by arrays of ``NaN`` values matching the event count.
    """
    RecordingDirectory = Path(RecordingDirectory)
    Spikes = np.load(RecordingDirectory / f"spikes_ch{Channel}.npy")
    FramePath = RecordingDirectory / f"frames_ch{Channel}.npy"
    LabelPath = RecordingDirectory / f"label_ch{Channel}.npy"
    Frames = np.load(FramePath).reshape(-1) if FramePath.exists() else np.full(Spikes.shape[0], np.nan)
    Labels = np.load(LabelPath).reshape(-1) if LabelPath.exists() else np.full(Spikes.shape[0], np.nan)
    return Spikes, Frames, Labels


def LoadSpikeRecording(RecordingDirectory):
    """Load every processed channel from one recording directory.

    Parameters
    ----------
    RecordingDirectory : path-like
        Directory containing ``spikes_ch*.npy`` and companion arrays.

    Returns
    -------
    dict
        Mapping from channel number to ``Spikes``, ``Frames``, and ``Labels``.
    """
    RecordingDirectory = Path(RecordingDirectory)
    Result = {}
    for SpikePath in sorted(RecordingDirectory.glob("spikes_ch*.npy")):
        Channel = ChannelFromPath(SpikePath)
        if Channel is not None:
            Spikes, Frames, Labels = LoadSpikeChannel(RecordingDirectory, Channel)
            Result[Channel] = {"Spikes": Spikes, "Frames": Frames, "Labels": Labels}
    return Result


def LoadSpikeFrames(RecordingDirectory, IncludeLabels=None):
    """Load sorted event frames grouped by channel.

    Parameters
    ----------
    RecordingDirectory : path-like
        Directory containing ``frames_ch*.npy`` arrays.
    IncludeLabels : iterable of int, optional
        Detector labels to retain when companion label arrays exist.

    Returns
    -------
    dict
        Mapping from channel number to sorted integer frame indices.

    Raises
    ------
    FileNotFoundError
        If the directory contains no channel frame arrays.
    """
    RecordingDirectory = Path(RecordingDirectory)
    Result = {}
    for FramePath in sorted(RecordingDirectory.glob("frames_ch*.npy")):
        Channel = ChannelFromPath(FramePath)
        if Channel is None:
            continue
        Frames = np.asarray(np.load(FramePath)).reshape(-1).astype(int)
        LabelPath = RecordingDirectory / f"label_ch{Channel}.npy"
        if IncludeLabels is not None and LabelPath.exists():
            Labels = np.asarray(np.load(LabelPath)).reshape(-1).astype(int)
            Length = min(len(Frames), len(Labels))
            Frames = Frames[:Length][np.isin(Labels[:Length], list(IncludeLabels))]
        Result[Channel] = np.sort(Frames)
    if not Result:
        raise FileNotFoundError(f"No frames_ch*.npy files found in {RecordingDirectory}.")
    return Result

def FindCorrelation(df, thresh=0.9, verbose=False):
    """
    Select a subset of variables by removing highly correlated features.

    The function computes the correlation matrix of the input DataFrame and
    iteratively removes variables whose correlation exceeds the specified
    threshold, returning the indices of the retained columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing the variables to analyze.
    thresh : float, optional
        Correlation threshold above which two variables are considered
        redundant. Default is 0.9.
    verbose : bool, optional
        If True, prints diagnostic information during the selection process.
        Default is False.

    Returns
    -------
    list
        Sorted list of column indices retained after correlation filtering.

    Raises
    ------
    ValueError
        If the input DataFrame contains only one variable.
    """
    CorrMatrix = df.corr()
    # corrMatrix.loc[:,:] =  np.triu(corrMatrix, k=0)

    Varnum = CorrMatrix.shape[0]

    if Varnum == 1:
        raise ValueError("only one variable given")

    # Re-order columns based on max absolute correlation
    '''
    original_order = np.arange(varnum)
    diag_mask = np.eye(varnum, dtype=bool)
    corrMatrix[diag_mask] = np.nan
    max_abs_corr_order = np.argsort(np.nanmax(np.abs(corrMatrix), axis=0))[::-1]
    corrMatrix = corrMatrix.iloc[:, max_abs_corr_order]
    new_order = original_order[max_abs_corr_order]
    temp_matrix = corrMatrix.copy()
    '''

    # Order columns based on average correlation
    OriginalOrder = np.arange(Varnum)
    DiagMask = np.eye(Varnum, dtype=bool)
    CorrMatrix[DiagMask] = np.nan
    MaxAbsCorrOrder = np.argsort(np.nanmax(-np.abs(CorrMatrix), axis=0))
    CorrMatrix = CorrMatrix.iloc[:, MaxAbsCorrOrder]
    NewOrder = OriginalOrder[MaxAbsCorrOrder]
    MeanAbsCorrOrder = np.argsort(np.nanmean(-np.abs(CorrMatrix), axis=0))
    CorrMatrix = CorrMatrix.iloc[:, MeanAbsCorrOrder]
    NewOrder = NewOrder[MeanAbsCorrOrder]
    TempMatrix = (CorrMatrix.copy())
    #temp_matrix[diag_mask] = np.nan
    #'''

    DeleteCol = list(OriginalOrder)
    OriginalOrder = list(OriginalOrder)
    NewOrder = list(NewOrder)
    Col = []
    Cont = 0
    while np.any(TempMatrix[~np.isnan(TempMatrix)] > thresh) and len(NewOrder)>0:
        Cont += 1
        T=time.time()
        # print('cycle n°:' +str(cont))
        if verbose:
            print("All correlations <=", thresh)
            break
        Idx = np.where(np.array(TempMatrix[NewOrder[0]])>thresh)[0]
        for I in range(len(Idx)):
            DeleteCol.remove(OriginalOrder[Idx[I]])
            NewOrder.remove(OriginalOrder[Idx[I]])
        Col.append(NewOrder[0])
        DeleteCol.remove(NewOrder[0])
        NewOrder.remove(NewOrder[0])
        OriginalOrder = DeleteCol.copy()
        TempMatrix = TempMatrix[NewOrder]
        TempMatrix = TempMatrix.loc[DeleteCol]
        # print(str(time.time()-t))

    if TempMatrix.shape[0]>0:
        for I in range(TempMatrix.shape[0]):
            Col.append(TempMatrix.columns[I])

    return sorted(Col)

def ExtractWaveforms(Data, Frames, WindowSamples):
    """Extract fixed-width waveforms centred on event frames.

    Parameters
    ----------
    Data : array-like
        One-dimensional signal trace.
    Frames : array-like
        Integer event-frame indices.
    WindowSamples : int
        Number of samples in each output waveform.

    Returns
    -------
    numpy.ndarray
        Waveforms with shape ``(len(Frames), WindowSamples)``. Samples beyond
        the signal boundaries are zero-padded.
    """
    Data = np.asarray(Data)
    Frames = np.asarray(Frames, dtype=int)
    HalfWindow = WindowSamples // 2
    Waveforms = np.zeros((len(Frames), WindowSamples), dtype=Data.dtype)
    for Index, Frame in enumerate(Frames):
        SourceStart = max(0, Frame - HalfWindow)
        SourceEnd = min(len(Data), Frame + HalfWindow + 1)
        DestinationStart = HalfWindow - (Frame - SourceStart)
        DestinationEnd = DestinationStart + SourceEnd - SourceStart
        Waveforms[Index, DestinationStart:DestinationEnd] = Data[SourceStart:SourceEnd]
    return Waveforms


def ResolveSpikePolarities(Data, NegativeFrames, PositiveFrames, HalfWindow):
    """Resolve nearby detections with opposite polarities.

    Parameters
    ----------
    Data : array-like
        Filtered signal used to compare event amplitudes.
    NegativeFrames : array-like
        Candidate negative-event frame indices.
    PositiveFrames : array-like
        Candidate positive-event frame indices.
    HalfWindow : int
        Maximum frame distance for treating two detections as one event.

    Returns
    -------
    tuple
        Sorted integer arrays ``(NegativeFrames, PositiveFrames)`` after the
        weaker member of each opposite-polarity pair has been removed.
    """
    Negative = set(int(Frame) for Frame in NegativeFrames)
    Positive = []
    for FrameP in PositiveFrames:
        FrameP = int(FrameP)
        Nearby = [Frame for Frame in Negative if abs(Frame - FrameP) <= HalfWindow]
        if not Nearby:
            Positive.append(FrameP)
            continue
        FrameN = min(Nearby, key=lambda Frame: abs(Frame - FrameP))
        if np.abs(Data[FrameN]) < np.abs(Data[FrameP]):
            Positive.append(FrameP)
            Negative.discard(FrameN)
    return np.asarray(sorted(Negative), dtype=int), np.asarray(sorted(Positive), dtype=int)


def DetectSpikesInTrace(
    Data,
    SamplingRate,
    NotchFrequency=50,
    LowCut=300,
    HighCut=3000,
    WindowSamples=41,
    DetectionThreshold=5,
):
    """Filter a trace and detect negative and positive events.

    Parameters
    ----------
    Data : array-like
        Raw one-dimensional channel trace.
    SamplingRate : float
        Sampling frequency in hertz.
    NotchFrequency : float, optional
        Frequency removed by the notch filter.
    LowCut : float, optional
        Lower band-pass cutoff in hertz.
    HighCut : float, optional
        Upper band-pass cutoff in hertz.
    WindowSamples : int, optional
        Samples extracted around each detected event.
    DetectionThreshold : float, optional
        Local standard-deviation multiplier used for peak detection.

    Returns
    -------
    tuple
        ``(Waveforms, Frames, Labels)`` where labels are zero for negative and
        one for positive detections.
    """
    Trace = np.asarray(Data, dtype=float)
    Trace = Trace - np.median(Trace)
    Trace = brw_f.NotchFilter(Trace, NotchFrequency, SamplingRate)
    Trace = brw_f.BandpassFilter(Trace, LowCut, HighCut, SamplingRate)
    Trace = Trace - np.mean(Trace)

    Step = int(SamplingRate * 0.05)
    NegativeFrames = SpikesDetection(Trace, Step, DetectionThreshold, "neg")
    PositiveFrames = SpikesDetection(Trace, Step, DetectionThreshold, "pos")
    NegativeFrames, PositiveFrames = ResolveSpikePolarities(
        Trace, NegativeFrames, PositiveFrames, WindowSamples // 2
    )

    NegativeWaveforms = ExtractWaveforms(Trace, NegativeFrames, WindowSamples)
    PositiveWaveforms = ExtractWaveforms(Trace, PositiveFrames, WindowSamples)
    Waveforms = np.concatenate([NegativeWaveforms, PositiveWaveforms], axis=0)
    Frames = np.concatenate([NegativeFrames, PositiveFrames])
    Labels = np.concatenate([
        np.zeros(len(NegativeFrames), dtype=int),
        np.ones(len(PositiveFrames), dtype=int),
    ])
    return Waveforms, Frames, Labels


def ProcessH5Channel(
    H5Path,
    OutputDirectory,
    WellID,
    Channel,
    ChannelRow,
    NotchFrequency=50,
    LowCut=300,
    HighCut=3000,
    WindowSamples=41,
    DetectionThreshold=5,
):
    """Detect and save events from one HDF5 trace column.

    Parameters
    ----------
    H5Path : path-like
        Input HDF5 file containing the trace matrix.
    OutputDirectory : path-like
        Destination for waveform, frame, and label NumPy arrays.
    WellID : str
        HDF5 well-group identifier.
    Channel : int
        Physical channel number used in output filenames.
    ChannelRow : int
        Column index of the channel in the HDF5 trace matrix.
    NotchFrequency : float, optional
        Frequency removed by the notch filter.
    LowCut : float, optional
        Lower band-pass cutoff in hertz.
    HighCut : float, optional
        Upper band-pass cutoff in hertz.
    WindowSamples : int, optional
        Samples extracted around each event.
    DetectionThreshold : float, optional
        Local standard-deviation multiplier for event detection.

    Returns
    -------
    None
        Results are saved as ``spikes_ch*``, ``frames_ch*``, and ``label_ch*``.
    """
    with h5py.File(H5Path, "r") as File:
        SamplingRate = float(File[WellID].attrs["SamplingRate"])
        Trace = np.asarray(File[WellID]["Traces"][:, ChannelRow], dtype=float)

    Waveforms, Frames, Labels = DetectSpikesInTrace(
        Trace,
        SamplingRate,
        NotchFrequency=NotchFrequency,
        LowCut=LowCut,
        HighCut=HighCut,
        WindowSamples=WindowSamples,
        DetectionThreshold=DetectionThreshold,
    )
    OutputDirectory = os.fspath(OutputDirectory)
    np.save(os.path.join(OutputDirectory, f"spikes_ch{Channel}.npy"), Waveforms)
    np.save(os.path.join(OutputDirectory, f"frames_ch{Channel}.npy"), Frames)
    np.save(os.path.join(OutputDirectory, f"label_ch{Channel}.npy"), Labels[:, np.newaxis])


def SpikesDetection(Data , step, threshold, aux_spike):
    """
    Detect the strongest threshold-crossing peak in each signal block.

    Parameters
    ----------
    Data : array-like
        One-dimensional filtered signal.
    step : int
        Number of samples per local detection block.
    threshold : float
        Local standard-deviation multiplier used as the peak threshold.
    aux_spike : str
        ``"pos"`` detects positive peaks; every other value detects negative
        peaks.

    Returns
    -------
    list
        Frame indices of the detected peaks.
    """
    Data =  Data[:Data .shape[0] - (Data .shape[0] % step)]
    DataReshaped = Data .reshape(-1, step)
    Mu = np.mean(DataReshaped,axis=1)
    Sigma = np.std(DataReshaped,axis=1)
    DataReshapedAux = DataReshaped-Mu[:, np.newaxis]
    ThSigma = threshold*Sigma

    Frames = []  # per salvare gli indici dei picchi per ogni riga
    for Ii in range(DataReshapedAux.shape[0]):
        Row = DataReshapedAux[Ii, :]
        if aux_spike == "pos":
            Peaks, Properties = find_peaks(Row, height=ThSigma[Ii])
        else:
            Peaks, Properties = find_peaks(-Row, height=ThSigma[Ii])
        if len(Peaks) > 0:
            Peaks = Peaks[np.argmax(Properties["peak_heights"])]
            Peaks  = Peaks + step*Ii
            Frames.append(Peaks)


    return Frames

def TemplateNeg(Data , ch, parameter = 4.5, algo = 'Leiden', distance = 'rho', method_HC = 'complete', criterion_HC = 'distance', method_KM = 'silhouette', max_iter_FCM=10, threshold_variance = 0.9, wMax  = 1, g = 1, epsilonEDR = 0.001, epsilonLCSS = 0.001, FuzzyParameter = 1, noise = 0, ThresholdDendrogram = 0.33, MaxClasses = [2], threshold_Leiden = 0.9, pMinkowski = 2, frequency=1000, Normalization = 'OFF', NormMode ='min_max_single'):
    """
    Negative templates learning

    Parameters
    ----------
    Data : np.ndarray
        2D matrix representing the Data set (number of frames x number of channels).
    ch : int
        Channel idx (between 0 and number of channels -1) on which we want to learn templates.
    parameter : float
        spikes detection parameter. Defaults to 4.5.
    algo : str
        clustering algorithm'. Defaults to 'Leiden'.
    distance : str
        metric for clustering. Defaults to 'rho'.
    method_HC : str
        linkage method. Defaults to 'complete'.
    criterion_HC : str
        hierarchical clustering criterion. Defaults to 'distance'.
    method_KM : str
        method to compute the optimal number of centroids in KM and FCM or relatives. Defaults to 'silhouette'.
    max_iter_FCM : int
        maximum number of iterations in FCM. Defaults to 10.
    threshold_variance : float
        explained variance after PCA. Defaults to 0.9.
    wMax : int
        WDTW and WDDTW parameter. Defaults to 1.
    g : int
        WDTW and WDDTW parameter. Defaults to 1.
    epsilonEDR : float
        EDR threshold. Defaults to 0.001.
    epsilonLCSS : float
        LCSS threshold. Defaults to 0.001.
    FuzzyParameter : int
        FCM parameter. Defaults to 1.
    noise : int
        amount of noise in percentage to add to Data . Defaults to 0.
    ThresholdDendrogram : float
        cut height of the dendrogram. Defaults to 0.33.
    MaxClasses : int
        maximum number of classes for clustering. Defaults to 1.
    threshold_Leiden : float
        Leiden threshold. Defaults to 0.9.
    pMinkowski : int
        Minkowski parameter. Defaults to 2.
    frequency : int
        STS parameter. Defaults to 1000.
    Normalization : str
        To applying Normalization. Defaults to 'OFF'.
    NormMode : str
        If Normalization applied, to select the modality. Defaults to 'min_max_single'.

    Returns
    -------
    clusters : list
        number of cluster and clusters obtained from template learning.
    templates_N : np.ndarray
        centroids of the clusters.
    frames_N : list
        frames idx of all the spikes detected.
    """

    DataChannel =  Data[:, ch].copy()
    NumFrames = DataChannel.shape[0]
    Step = int(frequency*0.01)
    T=0
    NFramesChNeg = 0
    FramesN = {}
    while(T<NumFrames-Step):
        Mu = np.mean(Data[T:T+Step, ch])
        Sigma = np.std(Data[T:T+Step, ch])
        FramesNeg = SpikesDetectionNeg(Data[T:T+Step,:] , ch, parameter)+T
        T=T+Step
        NFramesChNeg += len(FramesNeg)
        FramesN = set(FramesN)|set(FramesNeg)
    Mu = np.mean(Data[T:NumFrames, ch])
    Sigma = np.std(Data[T:NumFrames, ch])
    FramesNeg = SpikesDetectionNeg(Data[T:NumFrames,:] , ch, parameter)+T
    NFramesChNeg += len(FramesNeg)
    FramesN = set(FramesN)|set(FramesNeg)
    FramesN = sorted(FramesN)

    #'''
    T=0
    NFramesChPos = 0
    FramesP = {}
    while(T<=NumFrames-Step):
        Mu = np.mean(Data[T:T+Step, ch])
        Sigma = np.std(Data[T:T+Step, ch])
        FramesPos = SpikesDetectionNeg(- Data[T:T+Step,:] , ch, 4.5)+T
        T=T+Step
        NFramesChPos += len(FramesPos)
        FramesP = set(FramesP)|set(FramesPos)
    Mu = np.mean(Data[T:NumFrames, ch])
    Sigma = np.std(Data[T:NumFrames, ch])
    FramesPos = SpikesDetectionNeg(- Data[T:NumFrames,:] , ch, 4.5)+T
    NFramesChPos += len(FramesPos)
    FramesP = set(FramesP)|set(FramesPos)
    FramesP = sorted(FramesP)
    FramesNNew = set(FramesN)
    L=0
    while L < len(FramesP):
        if len(set(np.array(range(FramesP[L]-5, FramesP[L]))) & set(FramesN))>0:
            F = sorted(set(np.array(range(FramesP[L]-5, FramesP[L]))) & set(FramesN))[-1]
            if DataChannel[FramesP[L]]>-DataChannel[F]:
                FramesNNew.remove(F)
            L=L+1
        elif  len(set(np.array(range(FramesP[L], FramesP[L]+5+1))) & set(FramesN))>0:
            F = sorted(set(np.array(range(FramesP[L], FramesP[L]+5+1))) & set(FramesN))[0]
            if DataChannel[FramesP[L]]>-DataChannel[F]:
                FramesNNew.remove(F)
            L=L+1
        else:
            L=L+1
    FramesN = sorted(FramesNNew)
    #'''

    DatasetN = np.zeros((len(FramesN), 41))
    for K in range(len(FramesN)):
        PeakFrame = FramesN[K]
        if PeakFrame < 20:
            DatasetN[K, 20-PeakFrame:41] = DataChannel[0:PeakFrame+21]
        elif PeakFrame >= NumFrames-20:
            DatasetN[K, 0: NumFrames-PeakFrame+20] = DataChannel[PeakFrame-20:NumFrames]
        else:
            DatasetN[K] = DataChannel[PeakFrame-20:PeakFrame+21]
    DatasetNAux = DatasetN.copy()
    if DatasetNAux.shape[0]>1:
        Clusters = Stratification.RecursiveClustering(Data=DatasetNAux, Algo=algo, DistanceStr=distance, methodHC=method_HC, criterionHC=criterion_HC, methodKM=method_KM, MaxIterFCM=max_iter_FCM, ThresholdVariance=threshold_variance, wMax=wMax, g=g, epsilonEDR=epsilonEDR, epsilonLCSS=epsilonLCSS, FuzzyParameter=FuzzyParameter, noise=noise, ThresholdDendrogram=ThresholdDendrogram, MaxClasses=MaxClasses, ThresholdLeiden=threshold_Leiden, SamplingRate=frequency, pMinkowski=pMinkowski, Normalization=Normalization, NormMode=NormMode)

        TemplatesN =[]
        for C in range(Clusters[0]):
            Data = DatasetN[Clusters[1][C]]
            Data = Data .reshape((Data .shape[-2], Data .shape[-1]))
            Mu = np.mean(Data ,0)
            TemplatesN.append(Mu)
        TemplatesN = np.array(TemplatesN).T
        # '''
        Df = pd.DataFrame(TemplatesN)
        Corr = np.array(Df.corr())-np.eye(TemplatesN.shape[1])
        Idxs = set(np.arange(TemplatesN.shape[1]))
        while (np.max(Corr)>=0.95):
            IdxsDel = np.where(Corr==np.max(Corr))
            A = IdxsDel[0][0]
            B = IdxsDel[1][0]
            Idxs = Idxs-{A,B}
            ClustersNew =[]
            for I in Idxs:
                ClustersNew.append(Clusters[1][I])
            ClustersNew.append(list(set(Clusters[1][A])|set(Clusters[1][B])))
            Clusters = (len(ClustersNew), ClustersNew)
            TemplatesN =[]
            for C in range(Clusters[0]):
                Data = DatasetN[Clusters[1][C]]
                Data = Data .reshape((Data .shape[-2], Data .shape[-1]))
                Mu = np.mean(Data ,0)
                TemplatesN.append(Mu)
            TemplatesN = np.array(TemplatesN).T
            Df = pd.DataFrame(TemplatesN)
            Corr = np.array(Df.corr())-np.eye(TemplatesN.shape[1])
            Idxs = set(np.arange(TemplatesN.shape[1]))
            # '''

        '''
        plt.figure()
        for c in range(clusters[0]):
            # plt.figure()
            Data = Dataset_N[clusters[1][c]]
            mu = np.mean(Data ,0)
            sigma = np.std(Data ,0)
            plt.plot(np.arange(Data .shape[1]), mu)
            plt.fill_between(np.arange(Data .shape[1]), mu-sigma, mu+sigma, alpha = 0.2)
            # for k in range(len(clusters[1][c])):
                # plt.plot(np.arange(41), Dataset_N[clusters[1][c][k]])
            # plt.savefig('Cluster_'+str(c))
        plt.ylabel('(uV)')
        plt.xlabel('Frames')
        plt.title('3000 cells seeded, Day 97, chip 1003, channel 1210\nTemplates (PCA&k-means algo)')
        plt.savefig('Clusters_N')
        # '''

        ClustersN = []
        Templates = []
        FramesNew = ()
        Dt = 1/frequency
        for C in range(Clusters[0]):
            Dy = np.diff(TemplatesN[:,C])/Dt
            Dy = np.concatenate(([0], Dy))
            IPeaks = scipy.signal.find_peaks(-TemplatesN[:,C], width = 2.5)[0] #lavorare un po' qua
            # i_peaks = scipy.signal.find_peaks(-dy, height=200)[0]
            Der = Dy[IPeaks]
            # if len(scipy.signal.find_peaks(-templates_N[:,c], height = -(np.mean(templates_N[:,c])-0.5*np.std(templates_N[:,c])))[0])==1:
            #if len(der[der<-50])==1 and len(scipy.signal.find_peaks(-templates_N[:,c], height = -(np.mean(templates_N[:,c])))[0])==1 and len(clusters[1][c])>2:
            if len(scipy.signal.find_peaks(-TemplatesN[:,C], prominence=4)[0])==1 and len(Clusters[1][C])>2 and len(scipy.signal.find_peaks(TemplatesN[:,C], prominence=25)[0])<=1:
            # if len(clusters[1][c])>=len(frames_N)/25 and len(clusters[1][c])>=50:
            # if len(np.where(templates_N[i_peaks,c]<0)[0])==1:
                ClustersN.append(Clusters[1][C])
                Templates.append(TemplatesN[:, C])
                FramesNew = set(FramesNew)|set(np.array(FramesN)[Clusters[1][C]])


    elif DatasetNAux.shape[0]==0:
        ClustersN = []
        Templates = []
        FramesNew = []
    else:
        Clusters = (1, [[0]])
        TemplatesN = DatasetNAux.copy()
        ClustersN = []
        Templates = []
        FramesNew = ()
        Dt = 1/frequency
        for C in range(Clusters[0]):
            Dy = np.diff(TemplatesN[:,C])/Dt
            Dy = np.concatenate(([0], Dy))
            IPeaks = scipy.signal.find_peaks(-TemplatesN[:,C], width = 2.5)[0] #lavorare un po' qua
            Der = Dy[IPeaks]
            if len(scipy.signal.find_peaks(-TemplatesN[:,C], prominence=4)[0])==1 and len(Clusters[1][C])>2 and len(scipy.signal.find_peaks(TemplatesN[:,C], prominence=25)[0])<=1:
                ClustersN.append(Clusters[1][C])
                Templates.append(TemplatesN[:, C])
                FramesNew = set(FramesNew)|set(np.array(FramesN)[Clusters[1][C]])


    TemplatesN = np.array(Templates).T

    Clusters = [len(ClustersN), ClustersN]

    for Var in list(locals()):
        if Var != 'cluster' or Var != 'templates_N' or Var != 'frames_N':
            del locals()[Var]
    gc.collect()


    return Clusters, TemplatesN, FramesN #list(frames_new) #modificare in modo da avere solo i frames degli spikes che tengo

def TemplateMatching(Data , templates, thresh = 0.95):
    """
    Templates matching

    Parameters
    ----------
    Data : np.ndarray
        vector representing the signal where compute the templates matching.
    templates : np.ndarray
        matrix number of templates x number of frames representing found templates.
    thresh : float
        match when Pearson's correlation coefficient upper than the threshold. Defaults to 0.95.

    Returns
    -------
    frames : list
        frames idx associated to the matching template.
    Data : np.ndarray
        signal after matching subtraction.
    Dataset : list
        matching frames waveforms.
    DatasetIdx : list
        association between Data set element and matching templates.
    """

    N = templates.shape[0]
    if N==0:
        return [], Data , [], []
    else:
        Size = int(templates.shape[1]/2)
        X = set(np.arange(Data .shape[0]))
        X1 = set(np.arange(Size))
        X2 = set(np.arange(Size+4)+Data .shape[0]-Size-4) #5 generico
        Y = np.array(sorted(X-X1-X2))
        Frames = []
        DatasetIdx = []
        for C in range(N):
            Frames.append([])
            DatasetIdx.append([])
        Dataset = []
        '''Spikes detection by templates matching'''
        I = Y[0]
        while I <= Y[-1] :
            #t=time.time()
            Corr = []
            for C in range(N):
                Corr.append(pearsonr(templates[C]/np.linalg.norm(templates[C]),  Data[I-20:I+21]/np.linalg.norm(Data[I-20:I+21]))[0])
            Corr = np.array(Corr)
            if np.max(Corr)>thresh:
                Idx = np.where(Corr==np.max(Corr))[0][0]
                CorrAux = np.zeros(5)
                for J in range(5):
                    CorrAux[J]= pearsonr(templates[Idx]/np.linalg.norm(templates[Idx]),  Data[I+J-20:I+J+21]/np.linalg.norm(Data[I+J-20:I+J+21]))[0]
                JMax = np.where(CorrAux==np.max(CorrAux))[0][0]
                Frames[Idx].append(I+JMax)
                DatasetIdx[Idx].append(len(Dataset))
                Dataset.append(templates[Idx]/np.linalg.norm(templates[Idx])*np.linalg.norm(Data[I+JMax-20:I+JMax+21]))
                Data[I+JMax-20:I+JMax+21] = Data[I+JMax-20:I+JMax+21]-templates[Idx]/np.linalg.norm(templates[Idx])*np.linalg.norm(Data[I+JMax-20:I+JMax+21])
            else:
                I = I+1
            #print(str(i)+': '+str(time.time()-t))

        for Var in list(locals()):
            if Var != 'frames' or Var != 'Data ' or Var != 'Data set' or Var !='DatasetIdx':
                del locals()[Var]
        gc.collect()

        return Frames, Data , Dataset, DatasetIdx

def CrossCorrelogram(f_cluster_1, f_cluster_2, SamplingRate, NumFrames):
    """
    Compute and inspect the cross-correlogram between two spike trains.

    The function converts spike-frame indices into Neo SpikeTrain objects,
    computes the cross-correlation histogram (CCG), and compares the average
    activity around zero lag with activity in the side regions to detect
    potential refractory-period violations.

    Parameters
    ----------
    f_cluster_1 : array-like
        Spike-frame indices for the first cluster.
    f_cluster_2 : array-like
        Spike-frame indices for the second cluster.
    SamplingRate : float
        Acquisition sampling frequency in Hz.
    NumFrames : int
        Total number of frames in the recording.

    Notes
    -----
    The function currently prints a qualitative assessment of refractory
    behavior rather than returning a value. A low central peak relative to
    the side regions may indicate over-clustering.
    """
    SpikeTimes1 = f_cluster_1/SamplingRate
    SpikeTimes2 = f_cluster_2/SamplingRate

    TStop = NumFrames/SamplingRate
    StA = SpikeTrain(SpikeTimes1 * pq.s, t_stop=TStop * pq.s)
    StB = SpikeTrain(SpikeTimes1 * pq.s, t_stop=TStop * pq.s)
    Ccg, Bins = cross_correlation_histogram(StA, StB, window=41/SamplingRate* pq.s, bin_size=1/SamplingRate*1000*pq.ms, border_correction=False)

    CcgCounts = Ccg.magnitude.flatten()

    # Indici relativi ai bin centrali (±1 ms = ±10 bin)
    CenterBin = len(CcgCounts) // 2
    CenterRegion = CcgCounts[CenterBin - int(SamplingRate/1000) : CenterBin + int(SamplingRate/1000)+1]

    # Zona laterale (±3–5 ms)
    SideRegion = np.concatenate([
        CcgCounts[:int(SamplingRate/1000)],         # bin -5 ms a -4 ms
        CcgCounts[-int(SamplingRate/1000):]         # bin +4 a +5 ms
    ])

    MeanCenter = np.mean(CenterRegion)
    MeanSide = np.mean(SideRegion)

    print("Conteggio medio zona centrale:", MeanCenter)
    print("Conteggio medio zona laterale:", MeanSide)

    if MeanCenter < 0.25 * MeanSide:
        print("→ CCG refrattario: possibile overclustering")
    else:
        print("→ Nessuna chiara refrattarietà")

def ChannelSpksort(ch):
    """
    Retrieve the neighborhood channels around a reference channel.

    The function identifies all channels within a 5x5 spatial window centered
    on the specified channel and returns both the channel list and the
    position of the reference channel within that list.

    Parameters
    ----------
    ch : int
        Index of the reference channel on a 64-column electrode grid.

    Returns
    -------
    tuple
        Tuple containing:

        - chs : numpy.ndarray
            Sorted array of neighboring channel indices.
        - idx_ch : numpy.ndarray
            Index of the reference channel within ``chs``.
    """

    Row = ch//64
    Col = ch % 64
    Rows = np.arange(Row-2,Row+2+1)
    Cols = np.arange(Col-2, Col+2+1)
    Rows = Rows[Rows>=0]
    Cols = Cols[Cols>=0]
    Chs = []
    for I in Rows:
        for J in Cols:
            Chs.append(I*64+J)
    Chs = np.array(sorted(Chs))
    IdxCh = np.where(Chs==ch)[0]

    for Var in list(locals()):
        if Var != 'chs' or Var != 'idx_ch':
            del locals()[Var]
    gc.collect()

    return Chs, IdxCh

def LinkChsSpksort(results):
    """
    Estimate template overlap between neighboring channels.

    The function compares spike templates extracted from different channels
    using Pearson correlation and counts highly correlated template pairs.

    Parameters
    ----------
    results : list
        List containing spike sorting results for multiple channels.
        Each element is expected to contain the templates associated with
        a channel.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix where element (i, j) contains the number of template
        pairs with Pearson correlation greater than or equal to 0.95 between
        channels i and j.
    """
    NChs = len(results)
    CommonNeuron = np.zeros((NChs, NChs))
    for I in range(NChs):
        for J in np.arange(I+1, NChs):
            for S in range(len(results[I][0])):
                for K in range(len(results[J][0])):
                    if  pearsonr(results[I][0][S], results[J][0][K])[0]>=0.95:
                        CommonNeuron[I][J]+=1

    for Var in list(locals()):
        if Var != 'common_neuron':
            del locals()[Var]
    gc.collect()

    return CommonNeuron
