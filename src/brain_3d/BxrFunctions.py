"""
Codes with functions related to BXR file
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import h5py
import time
from . import BrwFunctions as brw_f
import json

def ReadBXR(Filename, WellID):
    """
    Read BXR file, return the BXR Data and print some information about the file:
    -file's name; 
    -Data and time of the recording; 
    -number of channels; 
    -length of the recording; 
    -number of Frames recorded; 
    -sampling frequency


    Args:
        Filename (str): name of the file and its extension .BXR 
        WellID (str): identifier of the selected well

    Returns:
        BXR (h5py): the BXR file in h5py
    """    
    BXR = h5py.File(Filename)

    Toc = np.array(BXR['TOC'])
    NumFrames = Toc[Toc.shape[0]-1,1]
    SamplingRate = BXR.attrs['SamplingRate']
    NumChannels = np.array(BXR[WellID + '/StoredChIdxs']).shape[0]
    Duration = NumFrames/SamplingRate

    print('--- File: ' + Filename + ' ---')
    print('Number of Channels: ' + str(NumChannels))
    print('File Duration: ' + str(Duration))
    print('Total Number of Frames: ' + str(NumFrames))
    print('Sampling Frequency: ' + str(SamplingRate) + ' Hz')
    print('---')

    return BXR

def ConversionTimeToFrames(BXR, Time):
    """
    Convert time in seconds to Frames based on sampling frequency.

    Args:
        BXR (BXRFile): BXR file object
        Time (float): time in seconds

    Returns:
        int: number of Frames corresponding to the time in seconds
    """    
    SamplingRate = BXR.attrs['SamplingRate']
    Frames = int(Time * SamplingRate)
    return Frames

def Spikes2Df(BXR, WellID, StartTime = 0, Duration = 0.05):
    """ 
    Selected a BXR file and a well, we read the Frames and the channel where
    spikes were detected (in a selected time interval)

    Args:
        BXR (BXRFile): BXR Data
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05

    Returns:
        SpikesFrames (array): It contains the Frames when spikes occured; if N channels have a spike at the frame T, then the frame T is repeated N times in the array
        SpikeChannels (array): It contains the channel that measured the spikes
    """    
    StartFrame = ConversionTimeToFrames(BXR, StartTime)
    EndFrame = StartFrame + ConversionTimeToFrames(BXR, Duration)
    SpikeFrames = np.array(BXR[WellID+'/SpikeTimes'])
    SpikeChannels = np.array(BXR[WellID+'/SpikeChIdxs'])
    Indexes = np.where((SpikeFrames>=StartFrame)&(SpikeFrames<=EndFrame))[0]
    return SpikeFrames[Indexes], SpikeChannels[Indexes]

def CleanSpikes(BXR, WellID, PercentageChannels):
    """
    Selected a BXR file, a well and a percentage of channels p, 
    we distinguish the Frames showing a number of spikes in a number
    of channel larger and smaller than
    the Threshold value (PercentageChannels)x(Number of channels)

    Args:
        BXR (BXRFile): file BXR opened from its path
        WellID (str): identifier of the selected well
        PercentageChannels (float): percentage of channels

    Returns:
        SpikesLower (list): contains the frames with a number of spikes smaller than the Threshold
        SpikesUpper (list): contains the frames  a number of spikes larger than the Threshold
    """    
    SpikeFrames = np.array(BXR[WellID+'/SpikeTimes'])
    SpikeChannels = np.array(BXR[WellID+'/SpikeChIdxs'])
    DifferentFrame = np.unique(SpikeFrames)
    SpikesLower = []
    SpikesUpper = []
    NumChannels = np.array(BXR[WellID + '/StoredChIdxs']).shape[0]
    Threshold = round(PercentageChannels*NumChannels/100)
    for T in DifferentFrame:
        Index = np.where(SpikeFrames == T)[0]
        Tupla = (T, SpikeChannels[Index])
        if len(Tupla[1]) < Threshold:
            SpikesLower.append(Tupla)
        else:
            SpikesUpper.append(Tupla)
    return  SpikesLower, SpikesUpper

def RasterPlot(BXR, WellID, StartTime=0, Duration=0.05):
    """Plot raster diagram of spike events in a selected time interval.

    Args:
        BXR (BXRFile): file BXR opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.05.

    Returns:
        None (displays matplotlib plot)
    """    
    SpikeFrames, SpikeChannels = Spikes2Df(BXR, WellID, StartTime, Duration)
    SamplingRate = BXR.attrs['SamplingRate']

    if len(SpikeFrames)==0:
        print('No spikes in the time interval ['+str(StartTime)+', '+str(StartTime+Duration)+']')
    else:

        NumChannels = np.array(BXR[WellID + '/StoredChIdxs']).shape[0]
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

def Burst2Df(BXR, WellID, StartTime = 0, Duration = 0.05):
    """
    Selected a BXR file and a well, we read when and where we have bursts in a selected time interval

    Args:
        BXR (BXRFile): file BXR opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second) I want to consider. Defaults to 0.05

    Returns:
        BurstFrames (array): a 1-dimensional array representing the time instant, in Frames, in which each spike burst has been detected
        BurstChannels (array): a 1-dimensional array representing for each detected spike burst the linear Index of the channel It has been recorded on
    """    
    BurstFrames = np.array(BXR[WellID+'/SpikeBurstTimes'])
    BurstChannels = np.array(BXR[WellID+'/SpikeBurstChIdxs'])
    StartFrame = ConversionTimeToFrames(BXR, StartTime)
    NumFrames = ConversionTimeToFrames(BXR, Duration)
    EndFrame = StartFrame+NumFrames

    Cont = 0
    I = 0
    while Cont == 0:
        if (StartFrame > BurstFrames[I][0]) & (StartFrame > BurstFrames[I][1]):
            I += 1
        else:
            Cont +=1
    
    Cont = 0
    J = 0
    while Cont == 0:
        if (EndFrame > BurstFrames[J][1]) & (EndFrame > BurstFrames[J+1][0]):
            J += 1
        else:
            Cont +=1
    
    if (EndFrame <= BurstFrames[0][0]) or (J-I) <0 :
        BurstFrames = []
        BurstChannels = []
    elif J-I == 0:
        BurstFrames = BurstFrames[I]
        BurstChannels = BurstChannels[I]
    else:
        BurstFrames = BurstFrames[I : J+1]
        BurstChannels = BurstChannels[I : J+1]
    return BurstFrames, BurstChannels

def BurstPlot(BXR, WellID, StartTime=0, Duration=0.01):
    """Plot burst events in a selected time interval with color-coded channels.

    Args:
        BXR (BXRFile): file BXR opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.01.

    Returns:
        None (displays matplotlib plot)
    """
    BurstFrames, BurstChannels = Burst2Df(BXR, WellID, StartTime, Duration)

    if len(BurstFrames) == 0:
        print('No bursts in the time interval ['+str(StartTime)+', '+str(StartTime+Duration)+']')
    else:
        #starting frame
        StartFrame = ConversionTimeToFrames(BXR, StartTime)
        #number of Frames considered
        NumFrames = ConversionTimeToFrames(BXR, Duration)
        EndFrame = StartFrame+NumFrames
        SamplingRate = BXR.attrs['SamplingRate']
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
                F1 = BurstFrames[0]
                F2 = BurstFrames[1]
                BurstTimesExtended.append(np.arange(F1,F2+1)/SamplingRate)
        else: 
            for It in np.arange(len(BurstFrames)):
                F1 = BurstFrames[It, 0]
                F2 = BurstFrames[It, 1]
                BurstTimesExtended.append(np.arange(F1,F2+1)/SamplingRate)
        NumChannels = np.array(BXR[WellID + '/StoredChIdxs']).shape[0]
        Count = 0
        Data = []
        for It in np.arange(NumChannels):
            Aux = np.where(BurstChannels == It)
            if len(Aux[0]) > 0:
                Tt = np.empty((0,))
                for ItAux in np.arange(len(Aux[0])):
                    AuxList = BurstTimesExtended[Aux[0][ItAux]]
                    Tt = np.concatenate((Tt,AuxList))
                Data.append(Tt)
                Count = Count+1
            #else:
            #    Data.append([])
        Colors1 = ['C{}'.format(I) for I in range(Count)]
        Fig, Ax = plt.subplots()
        plt.eventplot(Data, colors=Colors1)
        plt.title('Burst Plot, Time interval = ['+str(StartTime)+', '+str(StartTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('(channels)')
        plt.yticks(np.arange(Count))
        Ax.set_yticklabels(BurstChannelsUnique)
        plt.show()

def WaveformsPlot(BXR, WellID, StartTime=0, Duration=0.01, ChIdx=0):
    """Plot spike waveforms for a specific channel in a selected time interval.

    Args:
        BXR (BXRFile): file BXR opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.01.
        ChIdx (int, optional): channel Index to plot. Defaults to 0.

    Returns:
        None (displays matplotlib plot)
    """
    #starting frame
    StartFrame = ConversionTimeToFrames(BXR, StartTime)
    #number of Frames considered
    NumFrames = ConversionTimeToFrames(BXR, Duration)   
    # collect the TOCs
    Toc = np.array(BXR['TOC'])
    SpikeToc = np.array(BXR[WellID + '/SpikeTOC'])

    # collect experiment information
    MinDigitalValue = BXR.attrs['MinDigitalValue']
    MaxDigitalValue = BXR.attrs['MaxDigitalValue']
    MinAnalogValue = BXR.attrs['MinAnalogValue']
    MaxAnalogValue = BXR.attrs['MaxAnalogValue']
    DacFactor = (MaxAnalogValue - MinAnalogValue) / (MaxDigitalValue - MinDigitalValue)
    OffsetValue = MinAnalogValue - DacFactor * MinDigitalValue
    SamplingRate = BXR.attrs['SamplingRate']

    # from the given start position and duration (in Frames), find the corresponding range of spike positions using the TOC
    TocStartIdx = np.searchsorted(Toc[:, 1], StartFrame)
    TocEndIdx = min(np.searchsorted(Toc[:, 1], StartFrame + NumFrames, side='right') + 1, len(Toc) - 1)
    SpikeStartPosition = SpikeToc[TocStartIdx]
    SpikeEndPosition = SpikeToc[TocEndIdx]

    # collect the required spike Data
    SpikeDataTimestamps = BXR[WellID + '/SpikeTimes'][SpikeStartPosition:SpikeEndPosition]
    SpikeDataChIdxs = BXR[WellID + '/SpikeChIdxs'][SpikeStartPosition:SpikeEndPosition]

    SpikeSortingPerformed = BXR.__contains__(WellID + '/SpikeUnits')
    if SpikeSortingPerformed:
        SpikeDataChUnits = BXR[WellID + '/SpikeUnits'][SpikeStartPosition:SpikeEndPosition]

    WaveformLength = BXR[WellID + '/SpikeForms'].attrs['Wavelength']
    SpikeDataWaveforms = BXR[WellID + '/SpikeForms'][SpikeStartPosition*WaveformLength:SpikeEndPosition*WaveformLength]
    DataLength = SpikeEndPosition - SpikeStartPosition

    # collect the waveforms for the given time range and channel Index
    WaveformData = {} if SpikeSortingPerformed else []
    Ts = []
    for I in range(0, DataLength):
        if SpikeDataChIdxs[I] == ChIdx and StartFrame <= SpikeDataTimestamps[I] < StartFrame + NumFrames:
            Ts.append(SpikeDataTimestamps[I])
            if SpikeSortingPerformed:
                SpikeUnit = SpikeDataChUnits[I]
                if SpikeUnit not in WaveformData.keys():
                    WaveformData[SpikeUnit] = []
                WaveformData[SpikeUnit].append(SpikeDataWaveforms[I*WaveformLength:I*WaveformLength+WaveformLength])
            else:
                WaveformData.append(SpikeDataWaveforms[I*WaveformLength:I*WaveformLength+WaveformLength])
    
    # visualize waveforms for the given channel Index, if spike sorting was performed,
    # units will be plotted with different colors
    if len(WaveformData)==0:
        print('No waveforms for the channel '+str(ChIdx)+' in the time interval ['+str(StartTime)+', '+str(StartTime+Duration)+']')

    else:
        plt.figure()
        X = np.arange(0, WaveformLength, 1) / SamplingRate

        if SpikeSortingPerformed:
            Colors = list(mcolors.BASE_COLORS.keys())
            C = 0
            for Unit in WaveformData:
                for Waveform in WaveformData[Unit]:
                    # convert the waveform to analog
                    Y = OffsetValue + DacFactor * Waveform
                    plt.plot(X, Y, color=Colors[C])
                C += 1
        else:
            for Waveform in WaveformData:
                # convert the waveform to analog
                Y = OffsetValue + DacFactor * Waveform
                plt.plot(X, Y, color='blue')

        plt.title('Spike waveforms = '+str(len(Ts))+', channel = '+ str(ChIdx+1)+', '+'Time interval=['+str(StartTime)+', '+str(StartTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('(uV)')
        plt.legend()
        plt.show()

def FP2Df(BXR, WellID, StartTime = 0, Duration = 0.05):
    """
    Selected a BXR file and a well, we read when and where we have a FP in a selected time interval

    Args:
        BXR (BXRFile): file BXR opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second) I want to consider. Defaults to 0.05

    Returns:
        FPFrames (array): It contains the Frames when FPs occured; 
        FPChannels (array): It contains the channel that measured the FPs
    """    
    StartFrame = ConversionTimeToFrames(BXR, StartTime)
    EndFrame = StartFrame + ConversionTimeToFrames(BXR, Duration)
    FPFrames = np.array(BXR[WellID+'/FpTimes'])
    FPChannels = np.array(BXR[WellID+'/FpChIdxs'])
    Indexes = np.where((FPFrames>=StartFrame)&(FPFrames<=EndFrame))[0]
    return FPFrames[Indexes], FPChannels[Indexes]

def FPFormPlot(BXR, WellID, StartTime=0, Duration=0.05, ChIdx=0):
    """Plot false positive waveforms for a specific channel in a selected time interval.

    Args:
        BXR (BXRFile): file BXR opened from its path
        WellID (str): identifier of the selected well
        StartTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.05.
        ChIdx (int, optional): channel Index to plot. Defaults to 0.

    Returns:
        None (displays matplotlib plot)
    """  
    #starting frame
    StartFrame = ConversionTimeToFrames(BXR, StartTime)
    #number of Frames considered
    NumFrames = ConversionTimeToFrames(BXR, Duration)  
    # collect experiment information
    MinDigitalValue = BXR.attrs['MinDigitalValue']
    MaxDigitalValue = BXR.attrs['MaxDigitalValue']
    MinAnalogValue = BXR.attrs['MinAnalogValue']
    MaxAnalogValue = BXR.attrs['MaxAnalogValue']
    DacFactor = (MaxAnalogValue - MinAnalogValue) / (MaxDigitalValue - MinDigitalValue)
    OffsetValue = MinAnalogValue - DacFactor * MinDigitalValue
    SamplingRate = BXR.attrs['SamplingRate']
    FPForms = np.array(BXR[WellID+'/FpForms'])
    FPformLength = BXR[WellID + '/FpForms'].attrs['Wavelength']
    FPFrames, FPChannels = FP2Df(BXR, WellID, StartTime, Duration)
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
        X = np.arange(0, FPformLength, 1)/SamplingRate
        Colors = list(mcolors.XKCD_COLORS.keys())
        C = 0
        for I in Index:
            Y = OffsetValue + DacFactor * FPForms[I:I + FPformLength]
            plt.plot(X, Y, color=Colors[C])
            C += 1
        
        plt.title('FPforms = ' +str(len(Index))+ ', channel = '+ str(ChIdx+1)+', Time interval = ['+str(StartTime)+', '+str(StartTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('uV')
        plt.legend()
        plt.show()
    else: 
        print('No FP for the channel '+ str(ChIdx+1)+' in the time interval ['+str(StartTime)+', '+str(StartTime+Duration)+']')






