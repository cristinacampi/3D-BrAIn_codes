"""
Codes with functions related to BRW file
"""
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
        DownsamplingFrequency (float): chosen sampling frequency.
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

    Data, Frames2Save = ReadingRawData(brw, wellID, DownsamplingFrequency, StartTime, Duration)
    try:
        NumChannels = np.array(brw[wellID + '/StoredChIdxs']).shape[0]
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        NumChannels = len(info['CorePlateData']['StoredPlateElIdxs']) 
    AuxData = Data[:,row*NumChannels+col]

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
    try:
        SamplingRate = brw.attrs['SamplingRate']
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']
   
    StartFrame = int(SamplingRate * StartTime)
    EndFrame = int(SamplingRate *(StartTime+Duration))
   
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

def SingleChannelFramesWithPeaks(brw, wellID, DownsamplingFrequency, row, col, StartTime=0, Duration = 0.05, Threshold=0):
    """
    This function returns the frames for the channel define by row and colum
    where we have a peak larger than a threshold defined by the user
    
    Args:
        brw (BrwFile): file brw opened from its path.
        wellID (str): identifier of the selected well.
        DownsamplingFrequency (float): chosen sampling frequency.
        row (int): number from 0 to 63 that selects the row of the channel in the well.
        col (int): number from 0 to 63 that selects the column of the channel in the well.
        StartTime (float): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
        Threshold (float): the threshold on the level of activity. Defaults to 0.
    
    Returns:
        FramesWithPeaks (np.ndarray): array that contains the frames of the peaks.
    """
    y, Frames2Save = ReadingSingleChannel(brw, wellID, DownsamplingFrequency, row, col, StartTime, Duration)
    y = np.transpose(y)
    peaks = scipy.signal.find_peaks(y, threshold=Threshold)
    FramesWithPeaks = Frames2Save[peaks[0]]
    return FramesWithPeaks

def FramesWithPeaks(brw, wellID, DownsamplingFrequency, StartTime = 0, Duration = 0.05, Percentage = 0, Threshold=0):
    """
    This function returns the frames where we have peaks
    larger or lower than a threshold defined by the user
    
    Args:
        brw (BrwFile): file brw opened from its path.
        wellID (str): identifier of the selected well.
        DownsamplingFrequency (float): chosen sampling frequency.
        StartTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
        Percentage (int): Percentage of the channel with peaks. Defaults to 0.
        Threshold (float): the threshold on the level of activity. Defaults to 0.
    
    Returns:
        np.ndarray: (array): array that contains the frames of the peaks for all the channels.
        FramesUnderPerc (list): it contains the frames with a number of peaks smaller than Percentage*NumChannels.
        FramesOverPerc (list): it contains the frames with a number of peaks larger than Percentage*NumChannels.
    """
    Data, Frames2Save = ReadingRawData(brw, wellID, DownsamplingFrequency, StartTime, Duration)
    NumChannels = Data.shape[1]
    NumFrames2Save = len(Frames2Save)
    MatrixPeaks = np.zeros((NumFrames2Save, NumChannels))

    for ch in range(NumChannels):
        IndexPeaks = scipy.signal.find_peaks(Data[:,ch], threshold=Threshold)
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

def BRW2df(brw, wellID, DownsamplingFrequency, StartTime = 0, Duration = 0.05): 

    """
    Selected a BrwFile and a well, this function creates 2 dataframe:
    - one with the coordinates of the channels and
    - one with the evolution over time of the activity maps of the well
    
    Args:
        brw (BrwFile): file brw opened from its path.
        wellID (str): identifier of the selected well.
        DownsamplingFrequency (float): chosen sampling frequency.
        StartTime (float): starting time in seconds. Defaults to 0.
        Duration (float): Duration of the measurement (in second). Defaults to 0.05.
    
    Returns:
        DataFrame, DataFrame: -DfXY is a pandas dataframe where we saved the couples (X,Y) that indicate the coordinates of the channels (we read channels row by row). -DfAL is a pandas data frame where we saved frames and their respective activity maps vectorized like we read channels.
    """
    Dim1 = int(np.sqrt(np.array(brw[wellID + '/StoredChIdxs']).shape[0]))
    Dim2 = Dim1
    Data, Frames2Save = ReadingRawData(brw, wellID, DownsamplingFrequency, StartTime, Duration)

    ListaAL = []
    for it in np.arange(Data.shape[0]):
        aux = Data[it,:]
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
        DownsamplingFrequency (float): chosen sampling frequency.
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
    Data, Frames2Save = ReadingRawData(brw, wellID, SamplingRate, StartTime, Duration)
    spikesAL = np.zeros((Data.shape[0],Data.shape[1]))
    for i in range(len(SpikeFrames)):
        print('Spike at frame '+str(SpikeFrames[i])+', channel number '+str(SpikeChannels[i]+1))
        spikesAL[SpikeFrames[i]-StartFrame-1][SpikeChannels[i]] = Data[SpikeFrames[i]-StartFrame-1][SpikeChannels[i]] 
    return spikesAL  

def BandpassFilter(Data, Lowcut, Highcut, SamplingRate, nfilter=3, PercSamplingRate=0.5):
    """
    Band pass filter
    
    Args:
        Data (float): signals to be filtered.
        Lowcut (float): lower limit of the band frequency.
        Highcut (float): upper limit of the band frequency.
        SamplingRate (float): signal sampling rate.
        nfilter (int): the order of the filter. Defaults to 3.
        PercSamplingRate (float): factor for downsample. Defaults to 0.5.
    
    Returns:
        float: the filtered signal.
    """
    b,a = butter(nfilter, [Lowcut/(PercSamplingRate*SamplingRate), Highcut/(PercSamplingRate *SamplingRate)], btype = 'band' )
    filtered = filtfilt(b, a, Data)

    return filtered

def HighpassFilter(Data, Cut, SamplingRate, nfilter=3, PercSamplingRate=0.5):
    """
    High pass filter
    
    Args:
        Data (float): signals to be filtered.
        Cut (float): Frequency to remove from the signal.
        SamplingRate (float): signal sampling rate.
        nfilter (int): the order of the filter. Defaults to 3.
        PercSamplingRate (float): factor for downsample. Defaults to 0.5.
    
    Returns:
        float: the filtered signal.
    """
    b, a = butter(nfilter, Cut / (PercSamplingRate*SamplingRate), btype='high')
    filtered = filtfilt(b, a, Data)

    return filtered

def NotchFilter(Data, Cut, SamplingRate, qf=3):
    """
    Apply a notch filter to remove a specific frequency from the signal.
    
    Args:
        Data (float): signals to be filtered.
        Cut (float): Frequency to remove from the signal.
        SamplingRate (float): signal sampling rate.
        qf (int): Quality factor. Defaults to 3.
    
    Returns:
        float: the filtered signal.
    """
    b, a = iirnotch(Cut, qf, SamplingRate)
    filtered = filtfilt(b, a, Data)

    return filtered

def LowpassFilter(Data, Cut, SamplingRate, nfilter=3, PercSamplingRate=0.5):
    """
    Apply a low-pass filter to remove high-frequency components from the signal.
    
    Args:
        Data (float): signals to be filtered.
        Cut (float): Frequency to remove from the signal.
        SamplingRate (float): signal sampling rate.
        nfilter (int): the order of the filter. Defaults to 3.
        PercSamplingRate (float): factor for downsample. Defaults to 0.5.
    
    Returns:
        float: the filtered signal.
    """
    b, a = butter(nfilter, Cut / (PercSamplingRate*SamplingRate), btype='low')
    filtered = filtfilt(b, a, Data)

    return filtered

def CommonAverageReference(Data):
    """
    The median and then the mean are removed form the data
    
    Args:
        Data (float): signals to be transformed.
    
    Returns:
        float: the transformed signal.
    """
    median = np.median(Data, 1)
    Data = (Data.T - median).T
    mu = np.mean(Data,0)
    Data = Data - mu
    
    return Data

def WienerFilter(Data):
    """
    Wiener filter
    
    Args:
        Data (float): signals to be filtered.
    
    Returns:
        float: the filtered signal.
    """
    data = wiener(data)

    return data

def PercentileFilter(Data, percentile):
    """
    Apply a percentile filter to remove frequency components below a specified magnitude percentile.
    
    Args:
        Data (float): signals to be filtered.
        percentile (float): the magnitude percentile we want to remove from the data.
    
    Returns:
        float: the filtered signal.
    """
    Spectrum = np.fft.fft(Data)
    Magnitude = np.abs(Spectrum)
    Threshold = np.percentile(Magnitude, percentile)
    Spectrum[Magnitude < Threshold] = 0
    filtered = np.fft.ifft(Spectrum)

    return filtered 

def PlotlyGraph(Data, ch):
    """
    Plot a data channel to an HTML file using Plotly.
    
    Args:
        Data (float): data to be plotted.
        ch (int): sensor to be plotted.
    
    Returns:
        None: Generates an HTML file named 'Graph_channel_<ch>.html'.
    """
    df = pd.DataFrame({'x_axis': np.arange(Data.shape[0]), 'y_axis': Data[:,ch] })
    fig = px.line(df, x='x_axis', y='y_axis', title='Channel '+str(ch))
    fig.write_html('Graph_channel_'+str(ch)+'.html')   

