"""
Utilities for reading, filtering, and extracting metrics from BRW files.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pywt
import math
import scipy
from . import BxrFunctions
import time
from scipy.signal import find_peaks, butter, filtfilt, wiener, iirnotch
from statistics import median
import plotly.express as px
import json

def ReadBRW(Filename, WellID):
    """
    Open a BRW file and print basic recording information.

    Parameters
    ----------
    Filename : str
        Path to the BRW file.
    WellID : str
        Identifier of the selected well.

    Returns
    -------
    tuple
        Opened BRW file, sampling rate, number of channels, recording duration,
        and total number of frames.
    """
    BRW = h5py.File(Filename)

    Toc = np.array(BRW['TOC'])
    NumFrames = Toc[Toc.shape[0]-1,1]
    try:
        SamplingRate = BRW.attrs['SamplingRate']
        NumChannels = np.array(BRW[WellID + '/StoredChIdxs']).shape[0]
    except KeyError:
        Info = json.loads(BRW['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = Info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']
        NumChannels = len(Info['CorePlateData']['StoredPlateElIdxs'])

    Duration = NumFrames/SamplingRate

    print('--- File: ' + Filename + ' ---')
    print('Number of Channels: ' + str(NumChannels))
    print('File Duration: ' + str(Duration))
    print('Total Number of Frames: ' + str(NumFrames))
    print('Sampling Frequency: ' + str(SamplingRate) + ' Hz')
    print('---')

    return BRW, SamplingRate, NumChannels, Duration, NumFrames

def DecodeEventBasedRawData(BRW, Data, WellID, StartTime=0, Duration=0.05):
    """
    Decode event-based sparse raw data for a selected well and time interval.

    Parameters
    ----------
    BRW : h5py.File
        Opened BRW file.
    Data : dict
        Mapping from stored channel indices to initialized digital traces.
    WellID : str
        Identifier of the selected well.
    StartTime : float, optional
        Start time in seconds. Default is 0.
    Duration : float, optional
        Measurement duration in seconds. Default is 0.05.

    Returns
    -------
    dict
        Channel-indexed digital traces filled with decoded samples.
    """
    try:
        SamplingRate = BRW.attrs['SamplingRate']
    except KeyError:
        Info = json.loads(BRW['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = Info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']

    StartFrame = int(SamplingRate * StartTime)
    EndFrame = int(SamplingRate * (StartTime + Duration))

    # collect the TOCs
    Toc = np.array(BRW['TOC']) #dà errore con i dati vecchi (ho provato DataSet_02)
    if EndFrame < StartFrame:
        EndFrame = Toc[Toc.shape[0]-1,1]

    EventsToc = np.array(BRW[WellID + '/EventsBasedSparseRawTOC'])
    # from the given start position and Duration in frames, localize the corresponding event positions
    # using the TOC
    TocStartIdx = np.searchsorted(Toc[:, 1], StartFrame)
    TocEndIdx = min(np.searchsorted(Toc[:, 1], EndFrame, side='right')+ 1, len(Toc) - 1)
    EventsStartPosition = EventsToc[TocStartIdx]
    EventsEndPosition = EventsToc[TocEndIdx]
    # decode all data for the given well ID and time interval
    BinaryData = BRW[WellID + '/EventsBasedSparseRaw'][EventsStartPosition:EventsEndPosition]
    BinaryDataLength = len(BinaryData)
    Pos = 0
    while Pos < BinaryDataLength:
        ChIdx = int.from_bytes(BinaryData[Pos:Pos + 4], byteorder='little', signed=True)
        Pos += 4
        ChDataLength = int.from_bytes(BinaryData[Pos:Pos + 4], byteorder='little', signed=True)
        Pos += 4
        ChDataPos = Pos
        while Pos < ChDataPos + ChDataLength:
            FromInclusive = int.from_bytes(BinaryData[Pos:Pos + 8], byteorder='little', signed=True)
            Pos += 8
            ToExclusive = int.from_bytes(BinaryData[Pos:Pos + 8], byteorder='little', signed=True)
            Pos += 8
            RangeDataPos = Pos
            for J in range(FromInclusive, ToExclusive):
                if J >= EndFrame:
                    break
                if J >= StartFrame:
                    data[ChIdx][J - StartFrame] = int.from_bytes(BinaryData[RangeDataPos:RangeDataPos + 2], byteorder='little', signed=True)

                RangeDataPos += 2
            Pos += (ToExclusive - FromInclusive) * 2

    return Data

def ReadingRawData(BRW, WellID, DownsamplingFrequency, StartTime = 0, Duration = 0.05):
    """
    Read raw data from a BRW file for a specified time interval and downsampling frequency.

    Parameters
    ----------
    BRW : h5py.File
        Opened BRW file.
    WellID : str
        Identifier of the selected well.
    DownsamplingFrequency : float
        Target sampling frequency in Hz.
    StartTime : float, optional
        Start time in seconds. Default is 0.
    Duration : float, optional
        Measurement duration in seconds. Default is 0.05.

    Returns
    -------
    tuple
        Downsampled analog data and the corresponding frame indices.
    """
    try:
        SamplingRate = BRW.attrs['SamplingRate']
    except KeyError:
        Info = json.loads(BRW['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = Info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']

    StartFrame = int(SamplingRate * StartTime)
    EndFrame = int(SamplingRate *(StartTime+Duration))
    # collect experiment information
    try:
        MinDigitalValue = BRW.attrs['MinDigitalValue']
        MaxDigitalValue = BRW.attrs['MaxDigitalValue']
        MinAnalogValue = BRW.attrs['MinAnalogValue']
        MaxAnalogValue = BRW.attrs['MaxAnalogValue']
    except KeyError:
        Info = json.loads(BRW['ExperimentInfo'][()][0].decode('utf-8'))
        MinDigitalValue = Info['SignalConverter']['DigitalToAnalogConverter']['MinDigitalValue']
        MaxDigitalValue = Info['SignalConverter']['DigitalToAnalogConverter']['MaxDigitalValue']
        MinAnalogValue  = Info['SignalConverter']['DigitalToAnalogConverter']['MinAnalogValueMicroVolt']
        MaxAnalogValue  = Info['SignalConverter']['DigitalToAnalogConverter']['MaxAnalogValueMicroVolt']
    DacFactor = (MaxAnalogValue - MinAnalogValue) / (MaxDigitalValue - MinDigitalValue)
    OffsetValue = MinAnalogValue - DacFactor * MinDigitalValue

    Toc = np.array(BRW['TOC'])
    if EndFrame < StartFrame:
            EndFrame = Toc[Toc.shape[0]-1,1]

    try:
        ChIdxs = np.array(BRW[WellID + '/StoredChIdxs'])
    except KeyError:
        Info = json.loads(BRW['ExperimentInfo'][()][0].decode('utf-8'))
        ChIdxs = np.array(Info['CorePlateData']['StoredPlateElIdxs'])
    ChIdxs.sort()#
    NCh = len(ChIdxs)#
    NumChannels = ChIdxs.shape[0]

    if 'EventsBasedSparseRawTOC' in BRW[WellID]:
        DataDict = {}
        for ChIdx in ChIdxs:
            DataDict[ChIdx] = np.zeros(EndFrame-StartFrame, dtype=np.int16)
        DataDict = DecodeEventBasedRawData(BRW, DataDict, WellID, StartTime, Duration)

        Data = np.zeros((EndFrame-StartFrame, NCh))
        for D in range(NCh):
            Data[:, D] = np.array(DataDict[ChIdxs[D]], dtype=float)


    elif 'Raw' in BRW[WellID]:
        AuxData = BRW[WellID + '/Raw']
        AuxData = AuxData[StartFrame*NumChannels:EndFrame*NumChannels]
        Data = np.reshape(AuxData, (EndFrame-StartFrame, NumChannels))

    elif 'WaveletBasedEncodedRaw' in BRW[WellID]:
        CoefsTotalLength = len(BRW[WellID + '/WaveletBasedEncodedRaw'])
        CompressionLevel = BRW[WellID + '/WaveletBasedEncodedRaw'].attrs['CompressionLevel']
        FramesChunkLength = BRW[WellID + '/WaveletBasedEncodedRaw'].attrs['CompressionLevel']
        CoefsChunkLength = math.ceil(FramesChunkLength/pow(2, CompressionLevel))*2
        for ChIdx in ChIdxs:
            T = time.time()
            Data = []
            CoefsPosition = ChIdx * CoefsChunkLength
            while CoefsPosition < CoefsTotalLength:
                Coefs = BRW[WellID + '/WaveletBasedEncodedRaw'][CoefsPosition:CoefsPosition+CoefsChunkLength]
                Length = int(len(Coefs)/2)
                Frames = pywt.idwt(Coefs[:Length], Coefs[Length:], 'sym7', 'periodization')
                Length *= 2
                for I in range(1, CompressionLevel):
                    Frames = pywt.idwt(Frames[:Length], None, 'sym7', 'periodization')
                    Length *= 2
                Data.extend(Frames)
                CoefsPosition += CoefsChunkLength * NumChannels
            print(time.time()-T)
        BRW.close()

    Step = int(SamplingRate/DownsamplingFrequency)

    Frames2Save = np.arange(0, EndFrame-StartFrame, Step)
    AuxData = np.empty((len(Frames2Save), Data.shape[1]))
    for F in np.arange(len(Frames2Save)):
        if int(Frames2Save[F]) == EndFrame-StartFrame:
            AuxData[F, :] = Data[int(Frames2Save[F])-1, :]
        else:
            AuxData[F, :] = Data[int(Frames2Save[F]),:]

    Frames2Save = np.array(Frames2Save, dtype = int)

    AuxData = OffsetValue + DacFactor * AuxData

    return AuxData, Frames2Save+StartFrame

def ReadingSingleChannel(BRW, WellID, DownsamplingFrequency, row, col, StartTime = 0, Duration = 0.05):#to modify
    """
    Read the activity trace for one channel in a selected well.

    Parameters
    ----------
    BRW : h5py.File
        Opened BRW file.
    WellID : str
        Identifier of the selected well.
    DownsamplingFrequency : float
        Target sampling frequency in Hz.
    row : int
        Channel row index.
    col : int
        Channel column index.
    StartTime : float, optional
        Start time in seconds. Default is 0.
    Duration : float, optional
        Measurement duration in seconds. Default is 0.05.

    Returns
    -------
    tuple
        Selected channel signal and the corresponding frame indices.
    """

    Data, Frames2Save = ReadingRawData(BRW, WellID, DownsamplingFrequency, StartTime, Duration)
    try:
        NumChannels = np.array(BRW[WellID + '/StoredChIdxs']).shape[0]
    except KeyError:
        Info = json.loads(BRW['ExperimentInfo'][()][0].decode('utf-8'))
        NumChannels = len(Info['CorePlateData']['StoredPlateElIdxs'])
    AuxData = Data[:,row*NumChannels+col]

    return AuxData, Frames2Save

def PlotRawData(BRW, WellID, title, DownsamplingFrequency, row, col, StartTime=0, Duration=0.05):
    """
    Plot and save the activity trace for one channel in a selected well.

    Parameters
    ----------
    BRW : h5py.File
        Opened BRW file.
    WellID : str
        Identifier of the selected well.
    title : str
        Output PNG filename without extension.
    DownsamplingFrequency : float
        Target sampling frequency in Hz.
    row : int
        Channel row index.
    col : int
        Channel column index.
    StartTime : float, optional
        Start time in seconds. Default is 0.
    Duration : float, optional
        Measurement duration in seconds. Default is 0.05.
    """
    try:
        SamplingRate = BRW.attrs['SamplingRate']
    except KeyError:
        Info = json.loads(BRW['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = Info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']

    StartFrame = int(SamplingRate * StartTime)
    EndFrame = int(SamplingRate *(StartTime+Duration))

    Y, Frames2Save = ReadingSingleChannel(BRW, WellID, DownsamplingFrequency, row, col, StartTime, Duration)
    X = Frames2Save/SamplingRate
    Y = np.transpose(Y)

    plt.figure()
    plt.plot(X, Y, color="blue")
    plt.title('Raw Signal of the channel '+ str(row*64 + col) +', time interval = ['+str(round(StartFrame/SamplingRate*100)/100)+', '+str(round(EndFrame/SamplingRate*100)/100)+']')
    plt.xlabel('(sec)')
    plt.ylabel('(uV)')
    plt.savefig(title+".png")
    plt.show()

def SingleChannelFramesWithPeaks(BRW, WellID, DownsamplingFrequency, row, col, StartTime=0, Duration = 0.05, Threshold=0):
    """
    Return peak frame indices for one selected channel.

    Parameters
    ----------
    BRW : h5py.File
        Opened BRW file.
    WellID : str
        Identifier of the selected well.
    DownsamplingFrequency : float
        Target sampling frequency in Hz.
    row : int
        Channel row index.
    col : int
        Channel column index.
    StartTime : float, optional
        Start time in seconds. Default is 0.
    Duration : float, optional
        Measurement duration in seconds. Default is 0.05.
    Threshold : float, optional
        Peak detection threshold. Default is 0.

    Returns
    -------
    numpy.ndarray
        Frame indices where peaks are detected.
    """
    Y, Frames2Save = ReadingSingleChannel(BRW, WellID, DownsamplingFrequency, row, col, StartTime, Duration)
    Y = np.transpose(Y)
    Peaks = scipy.signal.find_peaks(Y, threshold=Threshold)
    FramesWithPeaks = Frames2Save[Peaks[0]]
    return FramesWithPeaks

def FramesWithPeaks(BRW, WellID, DownsamplingFrequency, StartTime = 0, Duration = 0.05, Percentage = 0, Threshold=0):
    """
    Detect peak frames across all channels in a selected time interval.

    Parameters
    ----------
    BRW : h5py.File
        Opened BRW file.
    WellID : str
        Identifier of the selected well.
    DownsamplingFrequency : float
        Target sampling frequency in Hz.
    StartTime : float, optional
        Start time in seconds. Default is 0.
    Duration : float, optional
        Measurement duration in seconds. Default is 0.05.
    Percentage : float, optional
        Fraction of channels used as the peak-count threshold. Default is 0.
    Threshold : float, optional
        Peak detection threshold. Default is 0.

    Returns
    -------
    tuple
        Binary peak matrix, frames below the channel-percentage threshold, and
        frames above or equal to the channel-percentage threshold.
    """
    Data, Frames2Save = ReadingRawData(BRW, WellID, DownsamplingFrequency, StartTime, Duration)
    NumChannels = Data.shape[1]
    NumFrames2Save = len(Frames2Save)
    MatrixPeaks = np.zeros((NumFrames2Save, NumChannels))

    for Ch in range(NumChannels):
        IndexPeaks = scipy.signal.find_peaks(Data[:,Ch], threshold=Threshold)
        MatrixPeaks[IndexPeaks[0], Ch] = 1

    NumPeaks = np.sum(MatrixPeaks, axis = 1)
    FramesOverPerc = []
    FramesUnderPerc = []
    PC = Percentage*NumChannels
    for T in range(NumFrames2Save):
        if NumPeaks[T]>=PC:
            FramesOverPerc.append(NumPeaks[T])
        else:
            FramesUnderPerc.append(NumPeaks[T])

    return  MatrixPeaks, FramesUnderPerc, FramesOverPerc

def BRW2df(BRW, WellID, DownsamplingFrequency, StartTime = 0, Duration = 0.05):

    """
    Convert BRW activity maps and channel coordinates to dataframes.

    Parameters
    ----------
    BRW : h5py.File
        Opened BRW file.
    WellID : str
        Identifier of the selected well.
    DownsamplingFrequency : float
        Target sampling frequency in Hz.
    StartTime : float, optional
        Start time in seconds. Default is 0.
    Duration : float, optional
        Measurement duration in seconds. Default is 0.05.

    Returns
    -------
    tuple
        Channel-coordinate dataframe and activity-map dataframe.
    """
    Dim1 = int(np.sqrt(np.array(BRW[WellID + '/StoredChIdxs']).shape[0]))
    Dim2 = Dim1
    Data, Frames2Save = ReadingRawData(BRW, WellID, DownsamplingFrequency, StartTime, Duration)

    ListAL = []
    for It in np.arange(Data.shape[0]):
        Aux = Data[It,:]
        TuplaAL = (int(Frames2Save[It]), Aux)
        ListAL.append(TuplaAL)

    ListXY  = []
    for It in np.arange(1,Dim1+1):
        TuplaXY = (It, np.arange(1,Dim2+1))
        ListXY.append(TuplaXY)

    DfXY = pd.DataFrame(ListXY, columns=["X", "Y"])
    DfAL = pd.DataFrame(ListAL, columns=["Frame", "Activity"])
    return DfXY, DfAL

def SpikesActivityLevel(BRW, bxr, WellID, DownsamplingFrequency, StartTime = 0, Duration = 0.05):
    """
    Build an activity-level matrix at detected spike frames.

    Parameters
    ----------
    BRW : h5py.File
        Opened BRW file.
    bxr : h5py.File
        Opened BXR file containing spike times and channels.
    WellID : str
        Identifier of the selected well.
    DownsamplingFrequency : float
        Target sampling frequency in Hz.
    StartTime : float, optional
        Start time in seconds. Default is 0.
    Duration : float, optional
        Measurement duration in seconds. Default is 0.05.

    Returns
    -------
    numpy.ndarray
        Matrix whose nonzero entries contain channel activity at spike frames.
    """
    try:
        SamplingRate = BRW.attrs['SamplingRate']
    except KeyError:
        Info = json.loads(BRW['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = Info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']
    StartFrame = int(SamplingRate * StartTime)
    SpikeFrames, SpikeChannels = BxrFunctions.Spikes2Df(bxr, WellID, StartTime, Duration)
    Data, Frames2Save = ReadingRawData(BRW, WellID, SamplingRate, StartTime, Duration)
    SpikesAL = np.zeros((Data.shape[0],Data.shape[1]))
    for I in range(len(SpikeFrames)):
        print('Spike at frame '+str(SpikeFrames[I])+', channel number '+str(SpikeChannels[I]+1))
        SpikesAL[SpikeFrames[I]-StartFrame-1][SpikeChannels[I]] = Data[SpikeFrames[I]-StartFrame-1][SpikeChannels[I]]
    return SpikesAL

def BandpassFilter(Data, Lowcut, Highcut, SamplingRate, nfilter=3, PercSamplingRate=0.5):
    """
    Apply a band-pass filter to a signal.

    Parameters
    ----------
    Data : array-like
        Signal to filter.
    Lowcut : float
        Lower cutoff frequency.
    Highcut : float
        Upper cutoff frequency.
    SamplingRate : float
        Signal sampling rate in Hz.
    nfilter : int, optional
        Filter order. Default is 3.
    PercSamplingRate : float, optional
        Nyquist scaling factor. Default is 0.5.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """
    B,A = butter(nfilter, [Lowcut/(PercSamplingRate*SamplingRate), Highcut/(PercSamplingRate *SamplingRate)], btype = 'band' )
    Filtered = filtfilt(B, A, Data)

    return Filtered

def HighpassFilter(Data, Cut, SamplingRate, nfilter=3, PercSamplingRate=0.5):
    """
    Apply a high-pass filter to a signal.

    Parameters
    ----------
    Data : array-like
        Signal to filter.
    Cut : float
        Cutoff frequency.
    SamplingRate : float
        Signal sampling rate in Hz.
    nfilter : int, optional
        Filter order. Default is 3.
    PercSamplingRate : float, optional
        Nyquist scaling factor. Default is 0.5.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """
    B, A = butter(nfilter, Cut / (PercSamplingRate*SamplingRate), btype='high')
    Filtered = filtfilt(B, A, Data)

    return Filtered

def NotchFilter(Data, Cut, SamplingRate, qf=3):
    """
    Apply a notch filter to remove a specific frequency from the signal.

    Parameters
    ----------
    Data : array-like
        Signal to filter.
    Cut : float
        Frequency to remove from the signal.
    SamplingRate : float
        Signal sampling rate in Hz.
    qf : int, optional
        Quality factor. Default is 3.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """
    B, A = iirnotch(Cut, qf, SamplingRate)
    Filtered = filtfilt(B, A, Data)

    return Filtered

def LowpassFilter(Data, Cut, SamplingRate, nfilter=3, PercSamplingRate=0.5):
    """
    Apply a low-pass filter to remove high-frequency components from the signal.

    Parameters
    ----------
    Data : array-like
        Signal to filter.
    Cut : float
        Cutoff frequency.
    SamplingRate : float
        Signal sampling rate in Hz.
    nfilter : int, optional
        Filter order. Default is 3.
    PercSamplingRate : float, optional
        Nyquist scaling factor. Default is 0.5.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """
    B, A = butter(nfilter, Cut / (PercSamplingRate*SamplingRate), btype='low')
    Filtered = filtfilt(B, A, Data)

    return Filtered

def CommonAverageReference(Data):
    """
    Apply common-average referencing after median subtraction.

    Parameters
    ----------
    Data : numpy.ndarray
        Signal matrix to transform.

    Returns
    -------
    numpy.ndarray
        Transformed signal matrix.
    """
    Median = np.median(Data, 1)
    Data = (Data.T - Median).T
    Mu = np.mean(Data,0)
    Data = Data - Mu

    return Data

def WienerFilter(Data):
    """
    Apply a Wiener filter to a signal.

    Parameters
    ----------
    Data : array-like
        Signal to filter.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """
    Data = wiener(Data)

    return Data

def PercentileFilter(Data, percentile):
    """
    Apply a percentile filter to remove frequency components below a specified magnitude percentile.

    Parameters
    ----------
    Data : array-like
        Signal to filter.
    percentile : float
        Magnitude percentile below which frequency components are removed.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """
    Spectrum = np.fft.fft(Data)
    Magnitude = np.abs(Spectrum)
    Threshold = np.percentile(Magnitude, percentile)
    Spectrum[Magnitude < Threshold] = 0
    Filtered = np.fft.ifft(Spectrum)

    return Filtered

def SpikesMetric(SamplingRate, NumFrames, SpikesN, SpikesP, DatasetN, DatasetP, Threshold=0.25):
    """
    Compute spike metrics for each channel and for the whole well.

    Negative and positive spike frames are merged using set union. For each
    unique spike frame, the corresponding waveform is retrieved from the
    negative or positive dataset and used to compute the peak-to-peak amplitude.

    If the same frame is present both in negative and positive spikes, the
    negative waveform is used first.

    Parameters
    ----------
    SamplingRate : float
        Acquisition sampling rate in Hz.
    NumFrames : int
        Total number of frames in the recording.
    SpikesN : list
        Negative spike frames for each channel.
    SpikesP : list
        Positive spike frames for each channel.
    DatasetN : list
        Negative spike waveforms for each channel.
    DatasetP : list
        Positive spike waveforms for each channel.
    Threshold : float, optional
        Minimum spike rate required to define an active electrode.
        Default is 0.25 Hz.

    Returns
    -------
    tuple
        Spike metrics including spike count, spike rate, ISI metrics,
        active electrode IDs, and peak-to-peak metrics.
    """

    NChs = len(SpikesN)

    SpikesCount = np.zeros(NChs)
    ISIMap = np.zeros(NChs)
    ISIVarianceMap = np.zeros(NChs)
    PeakToPeakMap = np.zeros(NChs)
    PeakToPeakStd = np.zeros(NChs)

    SpikesFramesWell = []
    PeakToPeakWell = []

    for Ch in range(NChs):
        SpikesFramesN = np.asarray(SpikesN[Ch])
        SpikesFramesP = np.asarray(SpikesP[Ch])
        SetN = set(SpikesFramesN)
        SetP = set(SpikesFramesP)
        SpikesFrames = np.array(sorted(SetN | SetP))
        PeakToPeakCh = np.zeros(len(SpikesFrames))
        if len(SpikesFrames) > 0:
            DatasetSpikes = []
            for Frame in SpikesFrames:
                if Frame in SetN:
                    FrameIndex = np.where(SpikesFramesN == Frame)[0][0]
                    DatasetSpikes.append(DatasetN[Ch][FrameIndex])
                elif Frame in SetP:
                    FrameIndex = np.where(SpikesFramesP == Frame)[0][0]
                    DatasetSpikes.append(DatasetP[Ch][FrameIndex])
            DatasetSpikes = np.asarray(DatasetSpikes)
            PeakToPeakCh = np.max(DatasetSpikes, axis=1) - np.min(DatasetSpikes, axis=1)
            SpikesFramesWell.append(SpikesFrames)
            PeakToPeakWell.append(PeakToPeakCh)
            PeakToPeakMap[Ch] = np.mean(PeakToPeakCh)
            if PeakToPeakMap[Ch] != 0:
                PeakToPeakStd[Ch] = np.std(PeakToPeakCh) / PeakToPeakMap[Ch]
            SpikesCount[Ch] = len(SpikesFrames)
            if len(SpikesFrames) > 1:
                ISICh = np.diff(SpikesFrames) / SamplingRate * 1000
                ISIMap[Ch] = np.mean(ISICh)
                if ISIMap[Ch] != 0:
                    ISIVarianceMap[Ch] = np.std(ISICh) / ISIMap[Ch]
        else:
            SpikesFramesWell.append(np.array([]))
            PeakToPeakWell.append(np.array([]))

    SpikesRate = SpikesCount / NumFrames * SamplingRate
    ActiveElectrodesMap = SpikesRate.copy()
    ActiveElectrodesID = np.where(ActiveElectrodesMap >= Threshold)[0]
    NonActiveElectrodesID = list(np.arange(NChs))
    ActiveElectrodesNumber = len(ActiveElectrodesID)
    SpikesFramesAE = set()
    PeakToPeakAE = []
    for Ch in ActiveElectrodesID:
        NonActiveElectrodesID.remove(Ch)
        SpikesFramesAE = SpikesFramesAE | set(SpikesFramesWell[Ch])
        for PeakToPeakValue in PeakToPeakWell[Ch]:
            PeakToPeakAE.append(PeakToPeakValue)
    SpikesCount[NonActiveElectrodesID] = 0
    PeakToPeakMap[NonActiveElectrodesID] = 0
    PeakToPeakStd[NonActiveElectrodesID] = 0
    SpikesRate[NonActiveElectrodesID] = 0
    ISIMap[NonActiveElectrodesID] = 0
    ISIVarianceMap[NonActiveElectrodesID] = 0
    SpikesFramesAE = np.array(sorted(SpikesFramesAE))
    if len(SpikesFramesAE) > 1:
        ISIWell = np.diff(SpikesFramesAE) / SamplingRate * 1000
        ISIWellAverage = np.mean(ISIWell)
        if ISIWellAverage != 0:
            ISIVarianceWell = np.std(ISIWell) / ISIWellAverage
        else:
            ISIVarianceWell = 0
    else:
        ISIWellAverage = 0
        ISIVarianceWell = 0

    NSpikesWell = np.sum(SpikesCount)
    SpikesWellRate = NSpikesWell / NumFrames * SamplingRate
    if len(PeakToPeakAE) > 0:
        PeakToPeakAverageWell = np.mean(PeakToPeakAE)
        if PeakToPeakAverageWell != 0:
            PeakToPeakStdWell = np.std(np.asarray(PeakToPeakAE)) / PeakToPeakAverageWell
        else:
            PeakToPeakStdWell = 0
    else:
        PeakToPeakAverageWell = 0
        PeakToPeakStdWell = 0

    return (
        SamplingRate,
        NumFrames,
        SpikesN,
        SpikesP,
        SpikesFramesAE,
        SpikesCount,
        SpikesRate,
        ISIMap,
        ISIVarianceMap,
        ISIWellAverage,
        ISIVarianceWell,
        NSpikesWell,
        SpikesWellRate,
        ActiveElectrodesID,
        ActiveElectrodesNumber,
        PeakToPeakMap,
        PeakToPeakStd,
        PeakToPeakAverageWell,
        PeakToPeakStdWell,
    )


def BurstsMetric(
    SamplingRate,
    NumFrames,
    SpikesN,
    SpikesP,
    SpikesFramesAE,
    SpikesCount,
    NSpikesWell,
    ActiveElectrodesID,
    NMinSpikes=5,
    ISIMaxSeconds=0.1,
):
    """
    Compute burst metrics for each active electrode and for the whole well.

    The function receives the outputs of ``SpikesMetric`` directly, instead of
    calling ``SpikesMetric`` internally. Bursts are detected as sequences of at
    least ``NMinSpikes`` spikes whose consecutive inter-spike intervals are not
    larger than ``ISIMaxSeconds``.

    Parameters
    ----------
    SamplingRate : float
        Acquisition sampling rate in Hz.
    NumFrames : int
        Total number of frames in the recording.
    SpikesN : list
        Negative spike frames for each channel.
    SpikesP : list
        Positive spike frames for each channel.
    SpikesFramesAE : numpy.ndarray
        Sorted unique spike frames across active electrodes.
    SpikesCount : numpy.ndarray
        Number of spikes for each channel.
    NSpikesWell : int or float
        Total number of spikes in active electrodes.
    ActiveElectrodesID : numpy.ndarray
        IDs of active electrodes.
    NMinSpikes : int, optional
        Minimum number of consecutive spikes required to define a burst.
        Default is 5.
    ISIMaxSeconds : float, optional
        Maximum inter-spike interval inside a burst, in seconds.
        Default is 0.1.

    Returns
    -------
    tuple
        Burst metrics for active electrodes and for the whole well.
    """
    Df = pd.DataFrame({'x_axis': np.arange(Data.shape[0]), 'y_axis': Data[:,ch] })
    Fig = px.line(Df, x='x_axis', y='y_axis', title='Channel '+str(ch))
    Fig.write_html('Graph_channel_'+str(ch)+'.html')   

def SpikesMetric(SamplingRate, NumFrames, SpikesN, SpikesP, DatasetN, DatasetP, Threshold=0.25):
    """
    Compute spike metrics for each channel and for the whole well.

    Negative and positive spike frames are merged using set union. For each
    unique spike frame, the corresponding waveform is retrieved from the
    negative or positive dataset and used to compute the peak-to-peak amplitude.

    If the same frame is present both in negative and positive spikes, the
    negative waveform is used first.

    Parameters
    ----------
    SamplingRate : float
        Acquisition sampling rate in Hz.
    NumFrames : int
        Total number of frames in the recording.
    SpikesN : list
        Negative spike frames for each channel.
    SpikesP : list
        Positive spike frames for each channel.
    DatasetN : list
        Negative spike waveforms for each channel.
    DatasetP : list
        Positive spike waveforms for each channel.
    Threshold : float, optional
        Minimum spike rate required to define an active electrode.
        Default is 0.25 Hz.

    Returns
    -------
    tuple
        Spike metrics including spike count, spike rate, ISI metrics,
        active electrode IDs, and peak-to-peak metrics.
    """

    NChs = len(SpikesN)

    SpikesCount = np.zeros(NChs)
    ISIMap = np.zeros(NChs)
    ISIVarianceMap = np.zeros(NChs)
    PeakToPeakMap = np.zeros(NChs)
    PeakToPeakStd = np.zeros(NChs)

    SpikesFramesWell = []
    PeakToPeakWell = []

    for Ch in range(NChs):
        SpikesFramesN = np.asarray(SpikesN[Ch])
        SpikesFramesP = np.asarray(SpikesP[Ch])
        SetN = set(SpikesFramesN)
        SetP = set(SpikesFramesP)
        SpikesFrames = np.array(sorted(SetN | SetP))
        PeakToPeakCh = np.zeros(len(SpikesFrames))
        if len(SpikesFrames) > 0:
            DatasetSpikes = []
            for Frame in SpikesFrames:
                if Frame in SetN:
                    FrameIndex = np.where(SpikesFramesN == Frame)[0][0]
                    DatasetSpikes.append(DatasetN[Ch][FrameIndex])
                elif Frame in SetP:
                    FrameIndex = np.where(SpikesFramesP == Frame)[0][0]
                    DatasetSpikes.append(DatasetP[Ch][FrameIndex])
            DatasetSpikes = np.asarray(DatasetSpikes)
            PeakToPeakCh = np.max(DatasetSpikes, axis=1) - np.min(DatasetSpikes, axis=1)
            SpikesFramesWell.append(SpikesFrames)
            PeakToPeakWell.append(PeakToPeakCh)
            PeakToPeakMap[Ch] = np.mean(PeakToPeakCh)
            if PeakToPeakMap[Ch] != 0:
                PeakToPeakStd[Ch] = np.std(PeakToPeakCh) / PeakToPeakMap[Ch]
            SpikesCount[Ch] = len(SpikesFrames)
            if len(SpikesFrames) > 1:
                ISICh = np.diff(SpikesFrames) / SamplingRate * 1000
                ISIMap[Ch] = np.mean(ISICh)
                if ISIMap[Ch] != 0:
                    ISIVarianceMap[Ch] = np.std(ISICh) / ISIMap[Ch]
        else:
            SpikesFramesWell.append(np.array([]))
            PeakToPeakWell.append(np.array([]))

    SpikesRate = SpikesCount / NumFrames * SamplingRate
    ActiveElectrodesMap = SpikesRate.copy()
    ActiveElectrodesID = np.where(ActiveElectrodesMap >= Threshold)[0]
    NonActiveElectrodesID = list(np.arange(NChs))
    ActiveElectrodesNumber = len(ActiveElectrodesID)
    SpikesFramesAE = set()
    PeakToPeakAE = []
    for Ch in ActiveElectrodesID:
        NonActiveElectrodesID.remove(Ch)
        SpikesFramesAE = SpikesFramesAE | set(SpikesFramesWell[Ch])
        for PeakToPeakValue in PeakToPeakWell[Ch]:
            PeakToPeakAE.append(PeakToPeakValue)
    SpikesCount[NonActiveElectrodesID] = 0
    PeakToPeakMap[NonActiveElectrodesID] = 0
    PeakToPeakStd[NonActiveElectrodesID] = 0
    SpikesRate[NonActiveElectrodesID] = 0
    ISIMap[NonActiveElectrodesID] = 0
    ISIVarianceMap[NonActiveElectrodesID] = 0
    SpikesFramesAE = np.array(sorted(SpikesFramesAE))
    if len(SpikesFramesAE) > 1:
        ISIWell = np.diff(SpikesFramesAE) / SamplingRate * 1000
        ISIWellAverage = np.mean(ISIWell)
        if ISIWellAverage != 0:
            ISIVarianceWell = np.std(ISIWell) / ISIWellAverage
        else:
            ISIVarianceWell = 0
    else:
        ISIWellAverage = 0
        ISIVarianceWell = 0

    NSpikesWell = np.sum(SpikesCount)
    SpikesWellRate = NSpikesWell / NumFrames * SamplingRate
    if len(PeakToPeakAE) > 0:
        PeakToPeakAverageWell = np.mean(PeakToPeakAE)
        if PeakToPeakAverageWell != 0:
            PeakToPeakStdWell = np.std(np.asarray(PeakToPeakAE)) / PeakToPeakAverageWell
        else:
            PeakToPeakStdWell = 0
    else:
        PeakToPeakAverageWell = 0
        PeakToPeakStdWell = 0

    return (
        SamplingRate,
        NumFrames,
        SpikesN,
        SpikesP,
        SpikesFramesAE,
        SpikesCount,
        SpikesRate,
        ISIMap,
        ISIVarianceMap,
        ISIWellAverage,
        ISIVarianceWell,
        NSpikesWell,
        SpikesWellRate,
        ActiveElectrodesID,
        ActiveElectrodesNumber,
        PeakToPeakMap,
        PeakToPeakStd,
        PeakToPeakAverageWell,
        PeakToPeakStdWell,
    )


def BurstsMetric(
    SamplingRate,
    NumFrames,
    SpikesN,
    SpikesP,
    SpikesFramesAE,
    SpikesCount,
    NSpikesWell,
    ActiveElectrodesID,
    NMinSpikes=5,
    ISIMaxSeconds=0.1,
):
    """
    Compute burst metrics for each active electrode and for the whole well.

    The function receives the outputs of ``SpikesMetric`` directly, instead of
    calling ``SpikesMetric`` internally. Bursts are detected as sequences of at
    least ``NMinSpikes`` spikes whose consecutive inter-spike intervals are not
    larger than ``ISIMaxSeconds``.

    Parameters
    ----------
    SamplingRate : float
        Acquisition sampling rate in Hz.
    NumFrames : int
        Total number of frames in the recording.
    SpikesN : list
        Negative spike frames for each channel.
    SpikesP : list
        Positive spike frames for each channel.
    SpikesFramesAE : numpy.ndarray
        Sorted unique spike frames across active electrodes.
    SpikesCount : numpy.ndarray
        Number of spikes for each channel.
    NSpikesWell : int or float
        Total number of spikes in active electrodes.
    ActiveElectrodesID : numpy.ndarray
        IDs of active electrodes.
    NMinSpikes : int, optional
        Minimum number of consecutive spikes required to define a burst.
        Default is 5.
    ISIMaxSeconds : float, optional
        Maximum inter-spike interval inside a burst, in seconds.
        Default is 0.1.

    Returns
    -------
    tuple
        Burst metrics for active electrodes and for the whole well.
    """

    NChs = len(SpikesN)
    ISIMaxFrames = int(SamplingRate * ISIMaxSeconds)

    Bursts = [[] for _ in range(NChs)]
    IBI = [[] for _ in range(NChs)]
    IBIAverage = []
    IBIStd = []

    BurstsDuration = [[] for _ in range(NChs)]
    BurstsNSpikes = [[] for _ in range(NChs)]
    BurstsDurationAverage = []
    BurstsNSpikesAverage = []
    BurstsISI = [[] for _ in range(NChs)]
    BurstsISIAverage = []
    NBursts = []
    ActiveElectrodesSet = set(ActiveElectrodesID)

    for Ch in range(NChs):
        if Ch in ActiveElectrodesSet:
            SpikesCh = np.array(sorted(set(SpikesN[Ch]) | set(SpikesP[Ch])))

            if len(SpikesCh) >= NMinSpikes:
                I = 0
                while I <= len(SpikesCh) - NMinSpikes:
                    IdxBurstStart = I
                    BurstStart = SpikesCh[I]
                    MaxBurstWindow = ISIMaxFrames * (NMinSpikes - 1)
                    if SpikesCh[I + NMinSpikes - 1] <= BurstStart + MaxBurstWindow:
                        ISIBursts = np.diff(SpikesCh[I:I + NMinSpikes])
                        if np.max(ISIBursts) <= ISIMaxFrames:
                            I = I + NMinSpikes - 1
                            while (
                                I < len(SpikesCh) - 1
                                and SpikesCh[I + 1] <= SpikesCh[I] + ISIMaxFrames
                            ):
                                I = I + 1
                            IdxBurstEnd = I
                            BurstFrames = SpikesCh[IdxBurstStart:IdxBurstEnd + 1]
                            Bursts[Ch].append(BurstFrames)
                            BurstsNSpikes[Ch].append(len(BurstFrames))
                            BurstsDuration[Ch].append(
                                (BurstFrames[-1] - BurstFrames[0]) / SamplingRate * 1000
                            )
                            BurstsISI[Ch].append(
                                ((BurstFrames[-1] - BurstFrames[0]) / (len(BurstFrames) - 1))
                                / SamplingRate
                                * 1000
                            )
                            I = I + 1
                        else:
                            I = I + 1
                    else:
                        I = I + 1

            NBursts.append(len(Bursts[Ch]))
            if NBursts[Ch] > 0:
                BurstsDurationAverage.append(np.mean(np.asarray(BurstsDuration[Ch])))
                BurstsNSpikesAverage.append(np.mean(np.asarray(BurstsNSpikes[Ch])))
                BurstsISIAverage.append(np.mean(np.asarray(BurstsISI[Ch])))
            else:
                BurstsDurationAverage.append(0)
                BurstsNSpikesAverage.append(0)
                BurstsISIAverage.append(0)
            if NBursts[Ch] > 1:
                for J in range(NBursts[Ch] - 1):
                    IBI[Ch].append((Bursts[Ch][J + 1][0] - Bursts[Ch][J][0]) / SamplingRate * 1000)
                IBIAverageCh = np.mean(np.asarray(IBI[Ch]))
                IBIAverage.append(IBIAverageCh)
                if IBIAverageCh != 0:
                    IBIStd.append(np.std(np.asarray(IBI[Ch])) / IBIAverageCh)
                else:
                    IBIStd.append(0)
            else:
                IBIAverage.append(0)
                IBIStd.append(0)
        else:
            NBursts.append(0)
            BurstsDurationAverage.append(0)
            BurstsNSpikesAverage.append(0)
            BurstsISIAverage.append(0)
            IBIAverage.append(0)
            IBIStd.append(0)
    NBursts = np.asarray(NBursts)
    BurstsRate = NBursts / NumFrames * SamplingRate * 60
    BurstsNSpikesAverage = np.asarray(BurstsNSpikesAverage)
    BurstsISIAverage = np.asarray(BurstsISIAverage)
    BurstsDurationAverage = np.asarray(BurstsDurationAverage)
    IBIAverage = np.asarray(IBIAverage)
    IBIStd = np.asarray(IBIStd)
    BurstsSpikesPercentage = np.zeros(NChs)

    if NSpikesWell > 0:
        for Ch in range(NChs):
            if SpikesCount[Ch] > 0 and NBursts[Ch] > 0:
                BurstsSpikesPercentage[Ch] = (
                    NBursts[Ch] * BurstsNSpikesAverage[Ch] / NSpikesWell * 100
                )
    BurstsWell = []
    BurstsDurationWell = []
    BurstsISIWell = []
    BurstsNSpikesWell = []
    I = 0
    while I <= len(SpikesFramesAE) - NMinSpikes:
        IdxBurstStart = I
        BurstStart = SpikesFramesAE[I]
        if SpikesFramesAE[I + NMinSpikes - 1] <= BurstStart + ISIMaxFrames * (NMinSpikes - 1):
            ISIBursts = np.diff(SpikesFramesAE[I:I + NMinSpikes])
            if np.max(ISIBursts) <= ISIMaxFrames:
                I = I + NMinSpikes - 1
                while (
                    I < len(SpikesFramesAE) - 1
                    and SpikesFramesAE[I + 1] <= SpikesFramesAE[I] + ISIMaxFrames
                ):
                    I = I + 1
                IdxBurstEnd = I
                BurstFrames = SpikesFramesAE[IdxBurstStart:IdxBurstEnd + 1]
                BurstsWell.append(BurstFrames)
                BurstsNSpikesWell.append(len(BurstFrames))
                BurstsDurationWell.append(
                    (BurstFrames[-1] - BurstFrames[0]) / SamplingRate * 1000
                )
                BurstsISIWell.append(
                    ((BurstFrames[-1] - BurstFrames[0]) / (len(BurstFrames) - 1))
                    / SamplingRate
                    * 1000
                )
                I = I + 1
            else:
                I = I + 1
        else:
            I = I + 1

    NBurstsWell = len(BurstsWell)
    BurstsRateWell = NBurstsWell / NumFrames * SamplingRate * 60
    BurstsNSpikesWell = np.asarray(BurstsNSpikesWell)
    BurstsDurationWell = np.asarray(BurstsDurationWell)
    BurstsISIWell = np.asarray(BurstsISIWell)
    if NBurstsWell > 0:
        BurstsNSpikesWellAverage = np.mean(BurstsNSpikesWell)
        BurstsDurationWellAverage = np.mean(BurstsDurationWell)
        BurstsISIWellAverage = np.mean(BurstsISIWell)
        BurstsNSpikesWellStd = (
            np.std(BurstsNSpikesWell) / BurstsNSpikesWellAverage
            if BurstsNSpikesWellAverage != 0
            else 0
        )
        BurstsDurationWellStd = (
            np.std(BurstsDurationWell) / BurstsDurationWellAverage
            if BurstsDurationWellAverage != 0
            else 0
        )
        BurstsISIWellStd = (
            np.std(BurstsISIWell) / BurstsISIWellAverage
            if BurstsISIWellAverage != 0
            else 0
        )
    else:
        BurstsNSpikesWellAverage = 0
        BurstsNSpikesWellStd = 0
        BurstsDurationWellAverage = 0
        BurstsDurationWellStd = 0
        BurstsISIWellAverage = 0
        BurstsISIWellStd = 0

    if NSpikesWell > 0:
        BurstsSpikesPercentageWell = (
            BurstsNSpikesWellAverage * NBurstsWell / NSpikesWell * 100
        )
    else:
        BurstsSpikesPercentageWell = 0

    IBIWell = []
    if NBurstsWell > 1:
        for J in range(NBurstsWell - 1):
            IBIWell.append(BurstsWell[J + 1][0] - BurstsWell[J][0])
    IBIWell = np.asarray(IBIWell) / SamplingRate * 1000

    if len(IBIWell) > 0:
        IBIWellAverage = np.mean(IBIWell)
        if IBIWellAverage != 0:
            IBIWellStd = np.std(IBIWell) / IBIWellAverage
        else:
            IBIWellStd = 0
    else:
        IBIWellAverage = 0
        IBIWellStd = 0

    return (
        Bursts,
        NBursts,
        BurstsRate,
        BurstsNSpikes,
        BurstsSpikesPercentage,
        BurstsISI,
        BurstsISIAverage,
        BurstsNSpikesAverage,
        BurstsDuration,
        BurstsDurationAverage,
        IBI,
        IBIAverage,
        IBIStd,
        BurstsWell,
        NBurstsWell,
        BurstsRateWell,
        BurstsDurationWellAverage,
        BurstsDurationWellStd,
        IBIWell,
        IBIWellAverage,
        IBIWellStd,
        BurstsNSpikesWell,
        BurstsNSpikesWellAverage,
        BurstsNSpikesWellStd,
        BurstsISIWell,
        BurstsISIWellAverage,
        BurstsISIWellStd,
        BurstsSpikesPercentageWell
    )


def NetworkBurstMetric(
    ActiveElectrodesID,
    Bursts,
    SamplingRate,
    NumFrames,
    PercentageAE=0.5,
    TimeWindowSeconds=0.25,
):
    """
    Compute network burst metrics from precomputed burst metrics.

    The function receives the outputs of ``BurstsMetric`` directly, instead of
    calling ``BurstsMetric`` internally. A network burst is detected when the
    number of active electrodes with at least one burst spike inside a time
    window is greater than or equal to ``PercentageAE`` times the number of
    active electrodes.

    Parameters
    ----------
    ActiveElectrodesID : numpy.ndarray
        IDs of active electrodes.
    Bursts : list
        Burst frame arrays for each channel.
    SamplingRate : float
        Acquisition sampling rate in Hz.
    NumFrames : int
        Total number of frames in the recording.
    PercentageAE : float, optional
        Fraction of active electrodes required to define a network burst.
        Default is 0.5.
    TimeWindowSeconds : float, optional
        Time window for network burst detection, in seconds.
        Default is 0.25.

    Returns
    -------
    tuple
        Network burst metrics: network burst counts per window, number of
        network bursts, network burst rate, duration, spike count, spike
        percentage, intra-network-burst ISI, inter-network-burst interval,
        and inter-network-burst interval standard deviation.
    """

    ActiveElectrodesID = np.asarray(ActiveElectrodesID)
    NActiveElectrodes = len(ActiveElectrodesID)

    TimeWindowFrames = int(TimeWindowSeconds * SamplingRate)

    if NActiveElectrodes == 0 or TimeWindowFrames <= 0:
        return (
            np.array([]),
            0,
            0,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            0,
        )

    BurstsSpikes = np.zeros((NActiveElectrodes, NumFrames))
    for ActiveIndex, Ch in enumerate(ActiveElectrodesID):
        for BurstFrames in Bursts[Ch]:
            BurstFrames = np.asarray(BurstFrames, dtype=int)
            BurstFrames = BurstFrames[(BurstFrames >= 0) & (BurstFrames < NumFrames)]
            BurstsSpikes[ActiveIndex, BurstFrames] = 1
    SpikesTot = np.sum(BurstsSpikes, axis=0)
    SpikesTot[SpikesTot > 0] = 1
    NSpikesTot = np.sum(SpikesTot)
    NWindows = int(NumFrames / TimeWindowFrames)
    if NWindows == 0:
        return (
            np.array([]),
            0,
            0,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            0,
        )

    TrimmedFrames = NWindows * TimeWindowFrames
    NBSpikesMatrix = SpikesTot[:TrimmedFrames].reshape(NWindows, TimeWindowFrames)
    NBSpikes = np.sum(NBSpikesMatrix, axis=1)
    NBDuration = np.zeros(NWindows)
    for I in range(NWindows):
        SpikeIndexes = np.where(NBSpikesMatrix[I] > 0)[0]
        if len(SpikeIndexes) > 1:
            Start = SpikeIndexes[0]
            End = SpikeIndexes[-1]
            NBDuration[I] = (End - Start) / SamplingRate
        else:
            NBDuration[I] = 0

    ActiveBurstMatrix = BurstsSpikes[:, :TrimmedFrames].reshape(
        NActiveElectrodes,
        NWindows,
        TimeWindowFrames,
    )

    Result = ActiveBurstMatrix.sum(axis=2)
    Result[Result > 0] = 1
    NetworkBurst = np.sum(Result, axis=0)
    IdxNB = np.where(NetworkBurst >= NActiveElectrodes * PercentageAE)[0]
    NNB = len(IdxNB)
    NBRate = NNB / NumFrames * SamplingRate
    if NSpikesTot > 0:
        NBSpikesPercentage = NBSpikes / NSpikesTot * 100
    else:
        NBSpikesPercentage = np.zeros_like(NBSpikes)
    NBISI = np.zeros_like(NBDuration)
    ValidNBSpikes = NBSpikes > 0
    NBISI[ValidNBSpikes] = NBDuration[ValidNBSpikes] * 1000 / NBSpikes[ValidNBSpikes]
    NBINBI = np.zeros(max(len(IdxNB) - 1, 0))

    for I in range(len(NBINBI)):
        StartIndexes = np.where(NBSpikesMatrix[IdxNB[I]] > 0)[0]
        EndIndexes = np.where(NBSpikesMatrix[IdxNB[I + 1]] > 0)[0]
        if len(StartIndexes) > 0 and len(EndIndexes) > 0:
            Start = StartIndexes[-1]
            End = EndIndexes[0] + TimeWindowFrames * (IdxNB[I + 1] - IdxNB[I])
            NBINBI[I] = (End - Start) / SamplingRate * 1000

    NBINBIVar = np.std(NBINBI) if len(NBINBI) > 0 else 0

    return (
        NetworkBurst,
        NNB,
        NBRate,
        NBDuration[IdxNB],
        NBSpikes[IdxNB],
        NBSpikesPercentage[IdxNB],
        NBISI[IdxNB],
        NBINBI,
        NBINBIVar,
    )
