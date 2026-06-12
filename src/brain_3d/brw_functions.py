"""
Codes with functions related to BRW file
"""
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pywt
import math
import scipy
import bxr_functions
import time
from scipy.signal import find_peaks, butter, filtfilt, wiener, iirnotch
from statistics import median
import plotly.express as px
import json

def ReadBRW(filename, wellID):
    """
    Read brw file, return the brw data and print some information about the file:
    -file's name;
    -data and time of the recording;
    -number of channels;
    -length of the recording;
    -number of frames recorded;
    -sampling frequency
    
    Args:
        filename (str): name of the file and its extension .brw.
        wellID (str): identifier of the selected well.
    
    Returns:
        brw (h5py): the brw data.
    """
    brw = h5py.File(filename)

    Toc = np.array(brw['TOC'])
    NumFrames = Toc[Toc.shape[0]-1,1] 
    try:
        SamplingRate = brw.attrs['SamplingRate']
        NumChannels = np.array(brw[wellID + '/StoredChIdxs']).shape[0]
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']
        NumChannels = len(info['CorePlateData']['StoredPlateElIdxs']) 

    Duration = NumFrames/SamplingRate

    print('--- File: ' + filename + ' ---')
    print('Number of Channels: ' + str(NumChannels))
    print('File Duration: ' + str(Duration))
    print('Total Number of Frames: ' + str(NumFrames))
    print('Sampling Frequency: ' + str(SamplingRate) + ' Hz')
    print('---')

    return brw, SamplingRate, NumChannels, Duration, NumFrames

def DecodeEventBasedRawData(brw, data, wellID, StartTime=0, Duration=0.05):
    """
    FROM 3BRAIN
    
    Args:
        brw (BrwFile): file brw opened from its path.
        data (dictionary): the keys are the recorded channel indexes StoredChIdxs and the values an array initialized with numFrames zeros for each key.
        wellID (str): identifier of the selected well.
        StartTime (float): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
    
    Returns:
        data (list): The returned data list contains digital samples that can be converted into analog values.
    """
    try:
        SamplingRate = brw.attrs['SamplingRate']
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']
        
    StartFrame = int(SamplingRate * StartTime)
    EndFrame = int(SamplingRate * (StartTime + Duration))
    
    # collect the TOCs
    Toc = np.array(brw['TOC']) #dà errore con i dati vecchi (ho provato DataSet_02)
    if EndFrame < StartFrame:
        EndFrame = Toc[Toc.shape[0]-1,1]
    
    EventsToc = np.array(brw[wellID + '/EventsBasedSparseRawTOC'])
    # from the given start position and Duration in frames, localize the corresponding event positions
    # using the TOC
    TocStartIdx = np.searchsorted(Toc[:, 1], StartFrame)
    TocEndIdx = min(np.searchsorted(Toc[:, 1], EndFrame, side='right')+ 1, len(Toc) - 1)
    EventsStartPosition = EventsToc[TocStartIdx]
    EventsEndPosition = EventsToc[TocEndIdx]
    # decode all data for the given well ID and time interval
    BinaryData = brw[wellID + '/EventsBasedSparseRaw'][EventsStartPosition:EventsEndPosition]
    BinaryDataLength = len(BinaryData)
    pos = 0
    while pos < BinaryDataLength:
        ChIdx = int.from_bytes(BinaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        ChDataLength = int.from_bytes(BinaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        ChDataPos = pos
        while pos < ChDataPos + ChDataLength:
            FromInclusive = int.from_bytes(BinaryData[pos:pos + 8], byteorder='little', signed=True)
            pos += 8
            ToExclusive = int.from_bytes(BinaryData[pos:pos + 8], byteorder='little', signed=True)
            pos += 8
            RangeDataPos = pos
            for j in range(FromInclusive, ToExclusive):
                if j >= EndFrame:
                    break
                if j >= StartFrame:
                    data[ChIdx][j - StartFrame] = int.from_bytes(BinaryData[RangeDataPos:RangeDataPos + 2], byteorder='little', signed=True)

                RangeDataPos += 2
            pos += (ToExclusive - FromInclusive) * 2

    return data 

def ReadingRawData(brw, wellID, DownsamplingFrequency, StartTime = 0, Duration = 0.05): 
    """
    Read raw data from a BRW file for a specified time interval and downsampling frequency.
    
    Args:
        brw (BrwFile): file brw opened from its path.
        wellID (str): identifier of the selected well.
        Downsampling_Frequency (float): chosen sampling frequency.
        StartTime (float): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
    
    Returns:
        AuxData (np.ndarray): array that contains the signals measured from StartFrame to EndFrame.
        Frames2Save (np.ndarray): array that contains the frames relative to measurements in AuxData.
    """
    try:
        SamplingRate = brw.attrs['SamplingRate']
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']
        
    StartFrame = int(SamplingRate * StartTime)
    EndFrame = int(SamplingRate *(StartTime+Duration))
    # collect experiment information
    try:
        MinDigitalValue = brw.attrs['MinDigitalValue']
        MaxDigitalValue = brw.attrs['MaxDigitalValue']
        MinAnalogValue = brw.attrs['MinAnalogValue']
        MaxAnalogValue = brw.attrs['MaxAnalogValue']
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        MinDigitalValue = info['SignalConverter']['DigitalToAnalogConverter']['MinDigitalValue']        
        MaxDigitalValue = info['SignalConverter']['DigitalToAnalogConverter']['MaxDigitalValue']        
        MinAnalogValue  = info['SignalConverter']['DigitalToAnalogConverter']['MinAnalogValueMicroVolt'] 
        MaxAnalogValue  = info['SignalConverter']['DigitalToAnalogConverter']['MaxAnalogValueMicroVolt'] 
    DacFactor = (MaxAnalogValue - MinAnalogValue) / (MaxDigitalValue - MinDigitalValue)
    OffsetValue = MinAnalogValue - DacFactor * MinDigitalValue

    Toc = np.array(brw['TOC'])
    if EndFrame < StartFrame:
            EndFrame = Toc[Toc.shape[0]-1,1]

    try:
        ChIdxs = np.array(brw[wellID + '/StoredChIdxs'])
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        ChIdxs = np.array(info['CorePlateData']['StoredPlateElIdxs'])
    ChIdxs.sort()#
    NCh = len(ChIdxs)#
    NumChannels = ChIdxs.shape[0]

    if 'EventsBasedSparseRawTOC' in brw[wellID]:
        DataDict = {}
        for ChIdx in ChIdxs:
            DataDict[ChIdx] = np.zeros(EndFrame-StartFrame, dtype=np.int16) 
        DataDict = DecodeEventBasedRawData(brw, DataDict, wellID, StartTime, Duration)

        data = np.zeros((EndFrame-StartFrame, NCh))
        for d in range(NCh):
            data[:, d] = np.array(DataDict[ChIdxs[d]], dtype=float)


    elif 'Raw' in brw[wellID]:
        AuxData = brw[wellID + '/Raw'] 
        AuxData = AuxData[StartFrame*NumChannels:EndFrame*NumChannels]
        data = np.reshape(AuxData, (EndFrame-StartFrame, NumChannels))

    elif 'WaveletBasedEncodedRaw' in brw[wellID]: 
        CoefsTotalLength = len(brw[wellID + '/WaveletBasedEncodedRaw'])
        CompressionLevel = brw[wellID + '/WaveletBasedEncodedRaw'].attrs['CompressionLevel']
        FramesChunkLength = brw[wellID + '/WaveletBasedEncodedRaw'].attrs['CompressionLevel']
        CoefsChunkLength = math.ceil(FramesChunkLength/pow(2, CompressionLevel))*2
        for ChIdx in ChIdxs:
            t = time.time()
            data = []
            coefsPosition = ChIdx * CoefsChunkLength
            while coefsPosition < CoefsTotalLength:
                coefs = brw[wellID + '/WaveletBasedEncodedRaw'][coefsPosition:coefsPosition+CoefsChunkLength]
                length = int(len(coefs)/2)
                frames = pywt.idwt(coefs[:length], coefs[length:], 'sym7', 'periodization') 
                length *= 2
                for i in range(1, CompressionLevel):
                    frames = pywt.idwt(frames[:length], None, 'sym7', 'periodization')
                    length *= 2
                data.extend(frames)
                coefsPosition += CoefsChunkLength * NumChannels
            print(time.time()-t) 
            print("un canale")   
        brw.close()


    Step = int(SamplingRate/DownsamplingFrequency)

    Frames2Save = np.arange(0, EndFrame-StartFrame, Step)
    AuxData = np.empty((len(Frames2Save), data.shape[1]))
    for f in np.arange(len(Frames2Save)):
        if int(Frames2Save[f]) == EndFrame-StartFrame:
            AuxData[f, :] = data[int(Frames2Save[f])-1, :]
        else:
            AuxData[f, :] = data[int(Frames2Save[f]),:]

    Frames2Save = np.array(Frames2Save, dtype = int)

    AuxData = OffsetValue + DacFactor * AuxData

    return AuxData, Frames2Save+StartFrame

def ReadingSingleChannel(brw, wellID, DownsamplingFrequency, row, col, StartTime = 0, Duration = 0.05):#to modify 
    """
    Selected a BRW file, a well and a channel in the well,
    this function reads the activity signal of the channel during the experiment
    
    Args:
        brw (BrwFile): file brw opened from its path.
        wellID (str): identifier of the selected well.
        DownsamplingFrequency (float): chosen sampling frequency.
        row (int): number from 0 to the maximum number of channel rows.
        col (int): number from 0 to the maximum number of channel columns.
        StartTime (float): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
    
    Returns:
        AuxData (np.ndarray): array that contains the signals measured in the selected channel.
        Frames2Save (np.ndarray): array that contains the frames relative to measurements in AuxData.
    """

    data, Frames2Save = ReadingRawData(brw, wellID, DownsamplingFrequency, StartTime, Duration)
    AuxData = data[:,row*data.shape[1]+col]
    return AuxData, Frames2Save

def PlotRawData(brw, wellID, title, DownsamplingFrequency, row, col, StartTime=0, Duration=0.05): 
    """
    Selected a BRW file, a well and a channel in the well,
    this function prints a graphic of the channel activity signal
    
    Args:
        brw (BrwFile): file brw opened from its path.
        wellID (str): identifier of the selected well.
        DownsamplingFrequency (float): chosen sampling frequency.
        row (int): number from 0 to 63 that selects the row of the channel in the well.
        col (int): number from 0 to 63 that selects the column of the channel in the well.
        StartTime (float): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
    """
    StartFrame = int(SamplingRate * StartTime)
    EndFrame = int(SamplingRate *(StartTime+Duration))
    try:
        SamplingRate = brw.attrs['SamplingRate']
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']
    y, Frames2Save = ReadingSingleChannel(brw, wellID, DownsamplingFrequency, row, col, StartTime, Duration)
    x = Frames2Save/SamplingRate
    y = np.transpose(y)

    plt.figure()
    plt.plot(x, y, color="blue")
    plt.title('Raw Signal of the channel '+ str(row*64 + col) +', time interval = ['+str(round(StartFrame/SamplingRate*100)/100)+', '+str(round(EndFrame/SamplingRate*100)/100)+']')
    plt.xlabel('(sec)')
    plt.ylabel('(uV)')
    plt.savefig(title+".png")
    plt.show()

def SingleChannelFramesWithPeaks(brw, wellID, Downsampling_Frequency, row, col, StartTime=0, Duration = 0.05, threshold=0):
    """
    This function returns the frames for the channel define by row and colum
    where we have a peak larger than a threshold defined by the user
    
    Args:
        brw (BrwFile): file brw opened from its path.
        wellID (str): identifier of the selected well.
        Downsampling_Frequency (float): chosen sampling frequency.
        row (int): number from 0 to 63 that selects the row of the channel in the well.
        col (int): number from 0 to 63 that selects the column of the channel in the well.
        StartTime (float): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
        threshold (float): the threshold on the level of activity. Defaults to 0.
    
    Returns:
        FramesWithPeaks (np.ndarray): array that contains the frames of the peaks.
    """
    y, Frames2Save = ReadingSingleChannel(brw, wellID, Downsampling_Frequency, row, col, StartTime, Duration)
    y = np.transpose(y)
    peaks = scipy.signal.find_peaks(y, threshold=threshold)
    FramesWithPeaks = Frames2Save[peaks[0]]
    return FramesWithPeaks

def FramesWithPeaks(brw, wellID, Downsampling_Frequency, StartTime = 0, Duration = 0.05, Percentage = 0, threshold=0):
    """
    This function returns the frames where we have peaks
    larger or lower than a threshold defined by the user
    
    Args:
        brw (BrwFile): file brw opened from its path.
        wellID (str): identifier of the selected well.
        Downsampling_Frequency (float): chosen sampling frequency.
        StartTime (c, optional): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
        Percentage (int): Percentage of the channel with peaks. Defaults to 0.
        threshold (float): the threshold on the level of activity. Defaults to 0.
    
    Returns:
        np.ndarray: (array): array that contains the frames of the peaks for all the channels.
        FramesUnderPerc (list): it contains the frames with a number of peaks smaller than Percentage*NumChannels.
        FramesOverPerc (list): it contains the frames with a number of peaks larger than Percentage*NumChannels.
    """
    data, Frames2Save = ReadingRawData(brw, wellID, Downsampling_Frequency, StartTime, Duration)
    NumChannels = data.shape[1]
    NumFrames2Save = len(Frames2Save)
    MatrixPeaks = np.zeros((NumFrames2Save, NumChannels))

    for ch in range(NumChannels):
        IndexPeaks = scipy.signal.find_peaks(data[:,ch], threshold=threshold)
        MatrixPeaks[IndexPeaks[0], ch] = 1

    NumPeaks = np.sum(MatrixPeaks, axis = 1)
    FramesOverPerc = []
    FramesUnderPerc = []
    PC = Percentage*NumChannels
    for t in range(NumFrames2Save):
        if NumPeaks[t]>=PC:
            FramesOverPerc.append(NumPeaks[t])
        else:
            FramesUnderPerc.append(NumPeaks[t])
    
    return  MatrixPeaks, FramesUnderPerc, FramesOverPerc

def BRW2df(brw, wellID, Downsampling_Frequency, StartTime = 0, Duration = 0.05): 

    """
    Selected a BrwFile and a well, this function creates 2 dataframe:
    - one with the coordinates of the channels and
    - one with the evolution over time of the activity maps of the well
    
    Args:
        brw (BrwFile): file brw opened from its path.
        wellID (str): identifier of the selected well.
        Downsampling_Frequency (float): chosen sampling frequency.
        StartTime (float): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
    
    Returns:
        DataFrame, DataFrame: -DfXY is a pandas dataframe where we saved the couples (X,Y) that indicate the coordinates of the channels (we read channels row by row). -DfAL is a pandas data frame where we saved frames and their respective activity maps vectorized like we read channels.
    """
    Dim1 = int(np.sqrt(np.array(brw[wellID + '/StoredChIdxs']).shape[0]))
    Dim2 = Dim1
    data, Frames2Save = ReadingRawData(brw, wellID, Downsampling_Frequency, StartTime, Duration)

    ListaAL = []
    for it in np.arange(data.shape[0]):
        aux = data[it,:]
        TuplaAL = (int(Frames2Save[it]), aux)
        ListaAL.append(TuplaAL)

    ListaXY  = []
    for it in np.arange(1,Dim1+1):
        TuplaXY = (it, np.arange(1,Dim2+1))
        ListaXY.append(TuplaXY)

    DfXY = pd.DataFrame(ListaXY, columns=["X", "Y"])
    DfAL = pd.DataFrame(ListaAL, columns=["Frame", "Activity"])
    return DfXY, DfAL

def SpikesActivityLevel(brw, bxr, wellID, DownsamplingFrequency, StartTime = 0, Duration = 0.05):
    """
    This function creates a matrix (number of frames saved x channels) where the entry ij is not zero if and only if
    the channel j has a spike at frame i; the entry is equal to the activity level of the channel j at frame i
    
    Args:
        brw (BrwFile): BRW file.
        bxr (BXRFile): BRW file.
        wellID (str): identifier of the selected well.
        Downsampling_Frequency (float): chosen sampling frequency.
        StartTime (float): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
    
    Returns:
        SpikesAL (np.ndarray): matrix where the entry ij is not zero if and only if the channel j has a spike at frame i.
    """
    try:
        SamplingRate = brw.attrs['SamplingRate']
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']
    StartFrame = int(SamplingRate * StartTime)
    SpikeFrames, SpikeChannels = bxr_functions.Spikes2Df(bxr, wellID, StartTime, Duration)
    data, Frames2Save = ReadingRawData(brw, wellID, SamplingRate, StartTime, Duration)
    spikesAL = np.zeros((data.shape[0],data.shape[1]))
    for i in range(len(SpikeFrames)):
        print('Spike at frame '+str(SpikeFrames[i])+', channel number '+str(SpikeChannels[i]+1))
        spikesAL[SpikeFrames[i]-StartFrame-1][SpikeChannels[i]] = data[SpikeFrames[i]-StartFrame-1][SpikeChannels[i]] 
    return spikesAL  

def BandpassFilter(data, lowcut, highcut, SamplingRate, nfilter=3, PercSamplingRate=0.5):
    """
    Band pass filter
    
    Args:
        data (float): signals to be filtered.
        lowcut (float): lower limit of the band frequency.
        highcut (float): upper limit of the band frequency.
        SamplingRate (float): signal sampling rate.
        nfilter (int): the order of the filter. Defaults to 3.
        PercSamplingRate (float): factor for downsample. Defaults to 0.5.
    
    Returns:
        float: the filtered signal.
    """
    b,a = butter(nfilter, [lowcut/(PercSamplingRate*SamplingRate), highcut/(PercSamplingRate *SamplingRate)], btype = 'band' )
    filtered = filtfilt(b, a, data)

    return filtered

def HighpassFilter(data, cut, SamplingRate, nfilter=3, PercSamplingRate=0.5):
    """
    High pass filter
    
    Args:
        data (float): signals to be filtered.
        cut (float): Frequency to remove from the signal.
        SamplingRate (float): signal sampling rate.
        nfilter (int): the order of the filter. Defaults to 3.
        PercSamplingRate (float): factor for downsample. Defaults to 0.5.
    
    Returns:
        float: the filtered signal.
    """
    b, a = butter(nfilter, cut / (PercSamplingRate*SamplingRate), btype='high')
    filtered = filtfilt(b, a, data)

    return filtered

def NotchFilter(data, cut, SamplingRate, qf=3):
    """
    Apply a notch filter to remove a specific frequency from the signal.
    
    Args:
        data (float): signals to be filtered.
        cut (float): Frequency to remove from the signal.
        SamplingRate (float): signal sampling rate.
        qf (int): Quality factor. Defaults to 3.
    
    Returns:
        float: the filtered signal.
    """
    b, a = iirnotch(cut, qf, SamplingRate)
    filtered = filtfilt(b, a, data)

    return filtered

def LowpassFilter(data, cut, SamplingRate, nfilter=3, PercSamplingRate=0.5):
    """
    Apply a low-pass filter to remove high-frequency components from the signal.
    
    Args:
        data (float): signals to be filtered.
        cut (float): Frequency to remove from the signal.
        SamplingRate (float): signal sampling rate.
        nfilter (int): the order of the filter. Defaults to 3.
        PercSamplingRate (float): factor for downsample. Defaults to 0.5.
    
    Returns:
        float: the filtered signal.
    """
    b, a = butter(nfilter, cut / (PercSamplingRate*SamplingRate), btype='low')
    filtered = filtfilt(b, a, data)

    return filtered

def CommonAverageReference(data):
    """
    The median and then the mean are removed form the data
    
    Args:
        data (float): signals to be tranformed.
    
    Returns:
        float: the trandferme signal.
    """
    median = np.median(data, 1)
    data = (data.T - median).T
    mu = np.mean(data,0)
    data = data - mu
    
    return data

def WienerFilter(data):
    """
    Wiener filter
    
    Args:
        data (float): signals to be filtered.
    
    Returns:
        float: the filtered signal.
    """
    data = wiener(data)

    return data

def PercentileFilter(data, percentile):
    """
    Apply a percentile filter to remove frequency components below a specified magnitude percentile.
    
    Args:
        data (float): signals to be filtered.
        percentile (float): the magnitude percentile we want to remove from the data.
    
    Returns:
        float: the filtered signal.
    """
    Spectrum = np.fft.fft(data)
    Magnitude = np.abs(Spectrum)
    Threshold = np.percentile(Magnitude, percentile)
    Spectrum[Magnitude < Threshold] = 0
    filtered = np.fft.ifft(Spectrum)

    return filtered 

def PlotlyGraph(data, ch):
    """
    Plot a data channel to an HTML file using Plotly.
    
    Args:
        data (float): data to be plotted.
        ch (int): sensor to be plotted.
    
    Returns:
        None: Generates an HTML file named 'Graph_channel_<ch>.html'.
    """
    df = pd.DataFrame({'x_axis': np.arange(data.shape[0]), 'y_axis': data[:,ch] })
    fig = px.line(df, x='x_axis', y='y_axis', title='Channel '+str(ch))
    fig.write_html('Graph_channel_'+str(ch)+'.html')   

