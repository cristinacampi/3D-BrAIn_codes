"""
Codes with functions related to BXR file
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
import h5py
import time
from . import brw_functions as brw_f

def ReadBXR(Filename, WellID):
    """
    Read Bxr file, return the Bxr Data and print some information about the file:
    -file's name; 
    -Data and time of the recording; 
    -number of channels; 
    -length of the recording; 
    -number of Frames recorded; 
    -sampling frequency


    Args:
        Filename (str): name of the file and its extension .Bxr 
        WellID (str): identifier of the selected well

    Returns:
        Bxr (h5py): the Bxr file in h5py
    """    
    Bxr = h5py.File(Filename)

    toc = np.array(Bxr['TOC'])
    NumFrames = toc[toc.shape[0]-1,1]
    SamplingRate = Bxr.attrs['SamplingRate']
    NumChannels = np.array(Bxr[WellID + '/StoredChIdxs']).shape[0]
    Duration = NumFrames/SamplingRate

    print('--- File: ' + Filename + ' ---')
    print('Number of Channels: ' + str(NumChannels))
    print('File Duration: ' + str(Duration))
    print('Total Number of Frames: ' + str(NumFrames))
    print('Sampling Frequency: ' + str(SamplingRate) + ' Hz')
    print('---')

    return Bxr

def ConversionTimeToFrames(Bxr, Time):
    """
    Convert time in seconds to Frames based on sampling frequency.

    Args:
        Bxr (BxrFile): BXR file object
        Time (float): time in seconds

    Returns:
        int: number of Frames corresponding to the time in seconds
    """    
    SamplingRate = Bxr.attrs['SamplingRate']
    Frames = int(Time * SamplingRate)
    return Frames

def Spikes2Df(Bxr, WellID, StartTime = 0, Duration = 0.05):
    """ 
    Selected a BXR file and a well, we read the Frames and the channel where
    spikes were detected (in a selected time interval)

    Args:
        Bxr (BXRFile): Bxr Data
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05

    Returns:
        SpikesFrames (array): It contains the Frames when spikes occured; if N channels have a spike at the frame T, then the frame T is repeated N times in the array
        SpikeChannels (array): It contains the channel that measured the spikes
    """    
    StartFrame = ConversionTimeToFrames(Bxr, StartTime)
    EndFrame = StartFrame + ConversionTimeToFrames(Bxr, Duration)
    SpikeFrames = np.array(Bxr[WellID+'/SpikeTimes'])
    SpikeChannels = np.array(Bxr[WellID+'/SpikeChIdxs'])
    Indexes = np.where((SpikeFrames>=StartFrame)&(SpikeFrames<=EndFrame))[0]
    return SpikeFrames[Indexes], SpikeChannels[Indexes]

def CleanSpikes(Bxr, WellID, PercentageChannels):
    """
    Selected a BXR file, a well and a percentage of channels p, 
    we distinguish the Frames showing a number of spikes in a number
    of channel larger and smaller than
    the Threshold value (PercentageChannels)x(Number of channels)

    Args:
        Bxr (BXRFile): file Bxr opened from its path
        WellID (str): identifier of the selected well
        PercentageChannels (float): percentage of channels

    Returns:
        SpikesLower (list): It contains the Frames with a number of spikes smaller than the Threshold
        SpikesUpper (list): It contains the Frames  a number of spikes larger than the Threshold
    """    
    SpikeFrames = np.array(Bxr[WellID+'/SpikeTimes'])
    SpikeChannels = np.array(Bxr[WellID+'/SpikeChIdxs'])
    DifferentFrame = np.unique(SpikeFrames)
    SpikesLower = []
    SpikesUpper = []
    NumChannels = np.array(Bxr[WellID + '/StoredChIdxs']).shape[0]
    Threshold = round(PercentageChannels*NumChannels/100)
    for t in DifferentFrame:
        Index = np.where(SpikeFrames == t)[0]
        Tupla = (t, SpikeChannels[Index])
        if len(Tupla[1]) < Threshold:
            SpikesLower.append(Tupla)
        else:
            SpikesUpper.append(Tupla)
    return  SpikesLower, SpikesUpper

def RasterPlot(Bxr, WellID, StartTime=0, Duration=0.05):
    """Plot raster diagram of spike events in a selected time interval.

    Args:
        Bxr (BxrFile): file Bxr opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.05.

    Returns:
        None (displays matplotlib plot)
    """    
    SpikeFrames, SpikeChannels = Spikes2Df(Bxr, WellID, StartTime, Duration)
    SamplingRate = Bxr.attrs['SamplingRate']

    if len(SpikeFrames)==0:
        print('No spikes in the time interval ['+str(StartTime)+', '+str(StartTime+Duration)+']')
    else:

        NumChannels = np.array(Bxr[WellID + '/StoredChIdxs']).shape[0]
        SpikeTimes = SpikeFrames/SamplingRate

        Data = []
        for It in np.arange(NumChannels):
            Aux = np.where(SpikeChannels==It)
            if len(Aux[0])>0:
                Tt = SpikeTimes[Aux]
                Data.append(Tt)
            else:
                Data.append([])

            
        plt.figure()
        plt.eventplot(Data, colors='black', lineoffsets=1, linelengths=2)
        plt.title('Spikes Raster Plot, Time interval = ['+str(StartTime)+', '+str(StartTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('(channels)')
        plt.show()

def Burst2Df(Bxr, WellID, StartTime = 0, Duration = 0.05):
    """
    Selected a BXR file and a well, we read when and where we have bursts in a selected time interval

    Args:
        Bxr (BXRFile): file Bxr opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second) I want to consider. Defaults to 0.05

    Returns:
        BurstFrames (array): a 1-dimensional array representing the time instant, in Frames, in which each spike burst has been detected
        BurstChannels (array): a 1-dimensional array representing for each detected spike burst the linear Index of the channel It has been recorded on
    """    
    BurstFrames = np.array(Bxr[WellID+'/SpikeBurstTimes'])
    BurstChannels = np.array(Bxr[WellID+'/SpikeBurstChIdxs'])
    #starting frame
    StartFrame = ConversionTimeToFrames(Bxr, StartTime)
    #number of Frames considered
    NumFrames = ConversionTimeToFrames(Bxr, Duration)
    EndFrame = StartFrame+NumFrames

    cont = 0
    i = 0
    while cont == 0:
        if (StartFrame > BurstFrames[i][0]) & (StartFrame > BurstFrames[i][1]):
            i += 1
        else:
            cont +=1
    
    cont = 0
    j = 0
    while cont == 0:
        if (EndFrame > BurstFrames[j][1]) & (EndFrame > BurstFrames[j+1][0]):
            j += 1
        else:
            cont +=1
    
    if (EndFrame <= BurstFrames[0][0]) or (j-i) <0 :
        BurstFrames = []
        BurstChannels = []
    elif j-i == 0:
        BurstFrames = BurstFrames[i]
        BurstChannels = BurstChannels[i]
    else:
        BurstFrames = BurstFrames[i : j+1]
        BurstChannels = BurstChannels[i : j+1]
    return BurstFrames, BurstChannels

def BurstPlot(Bxr, WellID, StartTime=0, Duration=0.01):
    """Plot burst events in a selected time interval with color-coded channels.

    Args:
        Bxr (BxrFile): file Bxr opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.01.

    Returns:
        None (displays matplotlib plot)
    """
    BurstFrames, BurstChannels = Burst2Df(Bxr, WellID, StartTime, Duration)

    if len(BurstFrames) == 0:
        print('No bursts in the time interval ['+str(StartTime)+', '+str(StartTime+Duration)+']')
    else:
        #starting frame
        StartFrame = ConversionTimeToFrames(Bxr, StartTime)
        #number of Frames considered
        NumFrames = ConversionTimeToFrames(Bxr, Duration)
        EndFrame = StartFrame+NumFrames
        SamplingRate = Bxr.attrs['SamplingRate']
        if len(BurstFrames)==1:
            if StartFrame > BurstFrames[0]:
                BurstFrames[0] = StartFrame
            if EndFrame < BurstFrames[1]:
                BurstFrames[1] = EndFrame
        else:
            if StartFrame > BurstFrames[0][0]:
                BurstFrames[0][0] = StartFrame
            if EndFrame < BurstFrames[len(BurstFrames)-1][1]:
                BurstFrames[len(BurstFrames)-1][1] = EndFrame
        BurstChannelsUnique = np.unique(BurstChannels)
        BurstTimesExtended = []
        if len(BurstFrames)==1:
            for It in np.arange(len(BurstFrames)):
                f1 = BurstFrames[0]
                f2 = BurstFrames[1]
                BurstTimesExtended.append(np.arange(f1,f2+1)/SamplingRate)
        else: 
            for It in np.arange(len(BurstFrames)):
                f1 = BurstFrames[It, 0]
                f2 = BurstFrames[It, 1]
                BurstTimesExtended.append(np.arange(f1,f2+1)/SamplingRate)
        NumChannels = np.array(Bxr[WellID + '/StoredChIdxs']).shape[0]
        count = 0
        Data = []
        for It in np.arange(NumChannels):
            Aux = np.where(BurstChannels == It)
            if len(Aux[0]) > 0:
                Tt = np.empty((0,))
                for ItAux in np.arange(len(Aux[0])):
                    AuxList = BurstTimesExtended[Aux[0][ItAux]]
                    Tt = np.concatenate((Tt,AuxList))
                Data.append(Tt)
                count = count+1
            #else:
            #    Data.append([])
        colors1 = ['C{}'.format(i) for i in range(count)]
        fig, ax = plt.subplots()
        plt.eventplot(Data, colors=colors1)
        plt.title('Burst Plot, Time interval = ['+str(StartTime)+', '+str(StartTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('(channels)')
        plt.yticks(np.arange(count))
        ax.set_yticklabels(BurstChannelsUnique)
        plt.show()

def WaveformsPlot(Bxr, WellID, StartTime=0, Duration=0.01, ChIdx=0):
    """Plot spike waveforms for a specific channel in a selected time interval.

    Args:
        Bxr (BxrFile): file Bxr opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.01.
        ChIdx (int, optional): channel Index to plot. Defaults to 0.

    Returns:
        None (displays matplotlib plot)
    """
    #starting frame
    StartFrame = ConversionTimeToFrames(Bxr, StartTime)
    #number of Frames considered
    NumFrames = ConversionTimeToFrames(Bxr, Duration)   
    # collect the TOCs
    toc = np.array(Bxr['TOC'])
    spikeToc = np.array(Bxr[WellID + '/SpikeTOC'])

    # collect experiment information
    MinDigitalValue = Bxr.attrs['MinDigitalValue']
    MaxDigitalValue = Bxr.attrs['MaxDigitalValue']
    MinAnalogValue = Bxr.attrs['MinAnalogValue']
    MaxAnalogValue = Bxr.attrs['MaxAnalogValue']
    dacFactor = (MaxAnalogValue - MinAnalogValue) / (MaxDigitalValue - MinDigitalValue)
    OffsetValue = MinAnalogValue - dacFactor * MinDigitalValue
    SamplingRate = Bxr.attrs['SamplingRate']
    ChIdxs = np.array(Bxr[WellID + '/StoredChIdxs'])

    # from the given start position and duration (in Frames), find the corresponding range of spike positions using the TOC
    TocStartIdx = np.searchsorted(toc[:, 1], StartFrame)
    TocEndIdx = min(np.searchsorted(toc[:, 1], StartFrame + NumFrames, side='right') + 1, len(toc) - 1)
    SpikeStartPosition = spikeToc[TocStartIdx]
    SpikeEndPosition = spikeToc[TocEndIdx]

    # collect the required spike Data
    SpikeDataTimestamps = Bxr[WellID + '/SpikeTimes'][SpikeStartPosition:SpikeEndPosition]
    SpikeDataChIdxs = Bxr[WellID + '/SpikeChIdxs'][SpikeStartPosition:SpikeEndPosition]

    SpikeSortingPerformed = Bxr.__contains__(WellID + '/SpikeUnits')
    if SpikeSortingPerformed:
        SpikeDataChUnits = Bxr[WellID + '/SpikeUnits'][SpikeStartPosition:SpikeEndPosition]

    WaveformLength = Bxr[WellID + '/SpikeForms'].attrs['Wavelength']
    SpikeDataWaveforms = Bxr[WellID + '/SpikeForms'][SpikeStartPosition*WaveformLength:SpikeEndPosition*WaveformLength]
    DataLength = SpikeEndPosition - SpikeStartPosition

    # collect the waveforms for the given time range and channel Index
    WaveformData = {} if SpikeSortingPerformed else []
    ts = []
    for i in range(0, DataLength):
        if SpikeDataChIdxs[i] == ChIdx and StartFrame <= spikeDataTimestamps[i] < StartFrame + NumFrames:
            ts.append(spikeDataTimestamps[i])
            if spikeSortingPerformed:
                spikeUnit = spikeDataChUnits[i]
                if spikeUnit not in WaveformData.keys():
                    WaveformData[spikeUnit] = []
                WaveformData[spikeUnit].append(SpikeDataWaveforms[i*waveformLength:i*waveformLength+waveformLength])
            else:
                WaveformData.append(SpikeDataWaveforms[i*waveformLength:i*waveformLength+waveformLength])
    
    # visualize waveforms for the given channel Index, if spike sorting was performed,
    # units will be plotted with different colors
    if len(WaveformData)==0:
        print('No waveforms for the channel '+str(ChIdx)+' in the time interval ['+str(StartTime)+', '+str(StartTime+Duration)+']')

    else:
        plt.figure()
        x = np.arange(0, WaveformLength, 1) / SamplingRate

        if spikeSortingPerformed:
            colors = list(mcolors.BASE_COLORS.keys())
            c = 0
            for unit in WaveformData:
                for waveform in WaveformData[unit]:
                    # convert the waveform to analog
                    y = OffsetValue + dacFactor * waveform
                    plt.plot(x, y, color=colors[c])
                c += 1
        else:
            for waveform in WaveformData:
                # convert the waveform to analog
                y = OffsetValue + dacFactor * waveform
                plt.plot(x, y, color='blue')

        plt.title('Spike waveforms = '+str(len(ts))+', channel = '+ str(ChIdx+1)+', '+'Time interval=['+str(StartTime)+', '+str(StartTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('(uV)')
        plt.legend()
        plt.show()

def FP2Df(Bxr, WellID, StartTime = 0, Duration = 0.05):
    """
    Selected a BXR file and a well, we read when and where we have a FP in a selected time interval

    Args:
        Bxr (BXRFile): file Bxr opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second) I want to consider. Defaults to 0.05

    Returns:
        FPFrames (array): It contains the Frames when FPs occured; 
        FPChannels (array): It contains the channel that measured the FPs
    """    
    StartFrame = ConversionTimeToFrames(Bxr, StartTime)
    EndFrame = StartFrame + ConversionTimeToFrames(Bxr, Duration)
    FPFrames = np.array(Bxr[WellID+'/FpTimes'])
    FPChannels = np.array(Bxr[WellID+'/FpChIdxs'])
    Indexes = np.where((FPFrames>=StartFrame)&(FPFrames<=EndFrame))[0]
    return FPFrames[Indexes], FPChannels[Indexes]

def FPFormPlot(Bxr, WellID, StartTime=0, Duration=0.05, ChIdx=0):
    """Plot false positive waveforms for a specific channel in a selected time interval.

    Args:
        Bxr (BxrFile): file Bxr opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.05.
        ChIdx (int, optional): channel Index to plot. Defaults to 0.

    Returns:
        None (displays matplotlib plot)
    """  
    #starting frame
    StartFrame = ConversionTimeToFrames(Bxr, StartTime)
    #number of Frames considered
    NumFrames = ConversionTimeToFrames(Bxr, Duration)  
    # collect experiment information
    MinDigitalValue = Bxr.attrs['MinDigitalValue']
    MaxDigitalValue = Bxr.attrs['MaxDigitalValue']
    MinAnalogValue = Bxr.attrs['MinAnalogValue']
    MaxAnalogValue = Bxr.attrs['MaxAnalogValue']
    dacFactor = (MaxAnalogValue - MinAnalogValue) / (MaxDigitalValue - MinDigitalValue)
    OffsetValue = MinAnalogValue - dacFactor * MinDigitalValue
    SamplingRate = Bxr.attrs['SamplingRate']
    FPForms = np.array(Bxr[WellID+'/FpForms'])
    FPformLength = Bxr[WellID + '/FpForms'].attrs['Wavelength']
    FPFrames, FPChannels = FP2Df(Bxr, WellID, StartTime, Duration)
    if StartFrame <= FPFrames[0]:
        FirstIndex = 0
    else:
        Fi = np.where(FPFrames>StartFrame)[0]
        FirstIndex = Fi[0]
    if StartFrame+NumFrames>=FPFrames[len(FPFrames)-1]:
        SecondIndex = len(FPFrames)-1
        Index = np.where(FPChannels[FirstIndex:SecondIndex+1]==ChIdx)[0]
    elif StartFrame+NumFrames<=FPFrames[0]:
        SecondIndex = 0
        Index = []
    else: 
        Si = np.where(FPFrames<=StartFrame+NumFrames)[0]
        SecondIndex = Si[len(Si)-1]
        Index = np.where(FPChannels[FirstIndex:SecondIndex+1]==ChIdx)[0]
    if len(Index)>0:    
        plt.figure()
        x = np.arange(0, FPformLength, 1)/SamplingRate
        colors = list(mcolors.XKCD_COLORS.keys())
        c = 0
        for i in Index:
            y = OffsetValue + dacFactor * FPForms[i:i + FPformLength]
            plt.plot(x, y, color=colors[c])
            c += 1
        
        plt.title('FPforms = ' +str(len(Index))+ ', channel = '+ str(ChIdx+1)+', Time interval = ['+str(StartTime)+', '+str(StartTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('uV')
        plt.legend()
        plt.show()
    else: 
        print('No FP for the channel '+ str(ChIdx+1)+' in the time interval ['+str(StartTime)+', '+str(StartTime+Duration)+']')

def SpikesDataset(brw, Bxr, WellID, DownsamplingFrequency, StartTime=0, Duration=0.05, ch=-10):
    """Generate a Dataset of time windows centered on spike events from a selected channel.

    Args:
        brw (BrwFile): file with raw Data
        Bxr (BxrFile): file with analyzed Data
        WellID (str): well identifier for the study
        DownsamplingFrequency (float): sampling frequency for downsampling
        StartTime (float, optional): start time in seconds. Defaults to 0.
        Duration (float, optional): duration of analysis window in seconds. Defaults to 0.05.
        ch (int, optional): channel ID. If negative, selects channel with highest spike count. Defaults to -10.

    Returns:
        np.ndarray: Dataset array where each row is a 40-frame window (based on sampling frequency) 
                    centered on a spike from the selected channel. Shape is [NumSpikes, window_length].
    """           
    #reading
    
    try:
        SamplingRate = brw.attrs['SamplingRate']
        NumChannels = np.array(brw[WellID + '/StoredChIdxs']).shape[0]
    except KeyError:
        info = json.loads(brw['ExperimentInfo'][()][0].decode('utf-8'))
        SamplingRate = info['SignalConverter']['SampleToTimeConverter']['FrameRateHertz']
        NumChannels = len(info['CorePlateData']['StoredPlateElIdxs']) 
    
    SpikeFrames, SpikeChannels = Spikes2Df(Bxr, WellID, StartTime, Duration)
    NumSpikes = len(SpikeFrames)

    SpikesForChannels = np.zeros(NumSpikes)
    for i in range(NumSpikes):
        ch = SpikeChannels[i]
        SpikesForChannels[ch] = SpikesForChannels[ch]+1
    max = np.max(SpikesForChannels)
    ChMax = np.where(SpikesForChannels == max)[0]
    if len(ChMax)==1:
        ChMax = ChMax
    else:
        ChMax = ChMax[0]

    if ch < 0:
        ChMax = ChMax
    else: 
        ChMax = ch

    t=time.time()
    DataFull_1m, FramesIndex1m = brw_f.ReadingRawData(brw, WellID, SamplingRate, 0, 60)
    print(time.time()-t)
    DataFull_2m, FramesIndex2m = brw_f.ReadingRawData(brw, WellID, SamplingRate, 60, 60)
    print(time.time()-t)
    DataFull_3m, FramesIndex3m = brw_f.ReadingRawData(brw, WellID, SamplingRate, 120, Duration-120)
    print(time.time()-t)
    

    F1m = len(FramesIndex1m)
    F2m = len(FramesIndex2m)
    F3m = len(FramesIndex3m)
    NumFrames = F1m + F2m + F3m
    # DataChannel = np.zeros((NumFrames,1))
    DataChannel = np.zeros(NumFrames)
    DataChannel[0:F1m] = DataFull_1m[:,ChMax]
    DataChannel[F1m:F1m+F2m] = DataFull_2m[:,ChMax]
    DataChannel[F2m+F1m:NumFrames] = DataFull_3m[:,ChMax]

    IdxChMax = np.where(SpikeChannels==ChMax)[0]
    SpikesChMax = SpikeFrames[IdxChMax]

    Data = []
    Frames = []

    step = round(SamplingRate/DownsamplingFrequency)

    for i in range(len(SpikesChMax)):
        CurrSpikeWindow = DataChannel[SpikesChMax[i]-20*step:SpikesChMax[i]+20*step+1]
        idx_NB = np.where(CurrSpikeWindow==-8000)[0]
        if len(idx_NB)==0:
            Data.append(CurrSpikeWindow)
            Frames.append(SpikesChMax[i])

    l = len(Data)
    w = len(Data[0])
    x = np.arange(0,w,step)
    Data = np.array(Data)
    # Dataset = np.zeros((l,w,1))
    Dataset = np.zeros((l,w))
    for i in range(l):
        Dataset[i] = Data[i]
    
    CurrFrame = Frames[0]
    Indexes = []
    Indexes.append(0)
    for i in range(l-1):
        if Frames[i+1]>=CurrFrame+40*step:
            Indexes.append(i+1)
            CurrFrame = Frames[i+1]

    Dataset = Dataset[Indexes]
    Frames = np.array(Frames)
    Frames = Frames[Indexes]
    SpikesPos = []
    SpikesNeg = []
    for i in range(len(Frames)):
        if DataChannel[Frames[i]]>0:
            SpikesPos.append(i)
        else:
            SpikesNeg.append(i)

    Dataset = np.reshape(Dataset, (Dataset.shape[0], Dataset.shape[1]))
    Dataset = Dataset[:, x]

    return Dataset





