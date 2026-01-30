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

def ReadBXR(filename, wellID):
    """
    Read bxr file, return the bxr data and print some information about the file:
    -file's name; 
    -data and time of the recording; 
    -number of channels; 
    -length of the recording; 
    -number of frames recorded; 
    -sampling frequency


    Args:
        filename (str): name of the file and its extension .bxr 
        wellID (str): identifier of the selected well

    Returns:
        bxr (h5py): the bxr file in h5py
    """    
    bxr = h5py.File(filename)

    toc = np.array(bxr['TOC'])
    NumFrames = toc[toc.shape[0]-1,1]
    SamplingRate= bxr.attrs['SamplingRate']
    NumChannels = np.array(bxr[wellID + '/StoredChIdxs']).shape[0]
    Duration = NumFrames/SamplingRate

    print('--- File: ' + filename + ' ---')
    print('Number of Channels: ' + str(NumChannels))
    print('File Duration: ' + str(Duration))
    print('Total Number of Frames: ' + str(NumFrames))
    print('Sampling Frequency: ' + str(SamplingRate) + ' Hz')
    print('---')

    return bxr

def ConversionTimeToFrames(bxr, Time):
    """Convert time in seconds to frames based on sampling frequency.

    Args:
        bxr (BxrFile): BXR file object
        Time (float): time in seconds

    Returns:
        int: number of frames corresponding to the time in seconds
    """    
    SamplingRate = bxr.attrs['SamplingRate']
    Frames = int(Time * SamplingRate)
    return Frames

def Spikes2df(bxr, wellID, startTime = 0, Duration = 0.05):
    """
    Selected a BXR file and a well, we read the frames and the channel where
    spikes were detected (in a selected time interval)

    Args:
        bxr (BXRFile): bxr data
        wellID (str): identifier of the selected well
        startTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05

    Returns:
        SpikesFrames (array): it contains the frames when spikes occured; 
            if N channels have a spike at the frame T, then the frame T is repeated N times in the array
        SpikeChannels (array): it contains the channel that measured the spikes
    """    
    startFrame = ConversionTimeToFrames(bxr, startTime)
    endFrame = startFrame + ConversionTimeToFrames(bxr, Duration)
    SpikeFrames = np.array(bxr[wellID+'/SpikeTimes'])
    SpikeChannels = np.array(bxr[wellID+'/SpikeChIdxs'])
    indexes = np.where((SpikeFrames>=startFrame)&(SpikeFrames<=endFrame))[0]
    return SpikeFrames[indexes], SpikeChannels[indexes]

def CleanSpikes(bxr, wellID, PercentageChannels):
    """
    Selected a BXR file, a well and a percentage of channels p, 
    we distinguish the frames showing a number of spikes in a number
    of channel larger and smaller than
    the threshold value (PercentageChannels)x(Number of channels)

    Args:
        bxr (BXRFile): file bxr opened from its path
        wellID (str): identifier of the selected well
        PercentageChannels (float): percentage of channels

    Returns:
        Spikes_lower (list): it contains the frames with a number of spikes smaller than the threshold
        Spikes_upper (list): it contains the frames  a number of spikes larger than the threshold
    """    
    SpikeFrames = np.array(bxr[wellID+'/SpikeTimes'])
    SpikeChannels = np.array(bxr[wellID+'/SpikeChIdxs'])
    DifferentFrame = np.unique(SpikeFrames)
    Spikes_lower = []
    Spikes_upper = []
    NumChannels = np.array(bxr[wellID + '/StoredChIdxs']).shape[0]
    threshold = round(PercentageChannels*NumChannels/100)
    for t in DifferentFrame:
        index = np.where(SpikeFrames == t)[0]
        tupla = (t, SpikeChannels[index])
        if len(tupla[1]) < threshold:
            Spikes_lower.append(tupla)
        else:
            Spikes_upper.append(tupla)
    return  Spikes_lower, Spikes_upper

def RasterPlot(bxr, wellID, startTime=0, Duration=0.05):
    """Plot raster diagram of spike events in a selected time interval.

    Args:
        bxr (BxrFile): file bxr opened from its path
        wellID (str): identifier of the selected well
        startTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.05.

    Returns:
        None (displays matplotlib plot)
    """    
    SpikeFrames, SpikeChannels = Spikes2df(bxr, wellID, startTime, Duration)
    SamplingRate = bxr.attrs['SamplingRate']

    if len(SpikeFrames)==0:
        print('No spikes in the time interval ['+str(startTime)+', '+str(startTime+Duration)+']')
    else:

        NumChannels = np.array(bxr[wellID + '/StoredChIdxs']).shape[0]
        SpikeTimes = SpikeFrames/SamplingRate

        data = []
        for it in np.arange(NumChannels):
            aux = np.where(SpikeChannels==it)
            if len(aux[0])>0:
                tt = SpikeTimes[aux]
                data.append(tt)
            else:
                data.append([])

            
        plt.figure()
        plt.eventplot(data, colors='black', lineoffsets=1, linelengths=2)
        plt.title('Spikes Raster Plot, Time interval = ['+str(startTime)+', '+str(startTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('(channels)')
        plt.show()

def Burst2df(bxr, wellID, startTime = 0, Duration = 0.05):
    """
    Selected a BXR file and a well, we read when and where we have bursts in a selected time interval

    Args:
        bxr (BXRFile): file bxr opened from its path
        wellID (str): identifier of the selected well
        startTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second) I want to consider. Defaults to 0.05

    Returns:
        BurstFrames (array): a 1-dimensional array representing the time instant, in frames, in which each spike burst has been detected
        BurstChannels (array): a 1-dimensional array representing for each detected spike burst the linear index of the channel it has been recorded on
    """    
    BurstFrames = np.array(bxr[wellID+'/SpikeBurstTimes'])
    BurstChannels = np.array(bxr[wellID+'/SpikeBurstChIdxs'])
    #starting frame
    startFrame = ConversionTimeToFrames(bxr, startTime)
    #number of frames considered
    NumFrames = ConversionTimeToFrames(bxr, Duration)
    endFrame = startFrame+NumFrames

    cont = 0
    i = 0
    while cont == 0:
        if (startFrame > BurstFrames[i][0]) & (startFrame > BurstFrames[i][1]):
            i += 1
        else:
            cont +=1
    
    cont = 0
    j = 0
    while cont == 0:
        if (endFrame > BurstFrames[j][1]) & (endFrame > BurstFrames[j+1][0]):
            j += 1
        else:
            cont +=1
    
    if (endFrame <= BurstFrames[0][0]) or (j-i) <0 :
        BurstFrames = []
        BurstChannels = []
    elif j-i == 0:
        BurstFrames = BurstFrames[i]
        BurstChannels = BurstChannels[i]
    else:
        BurstFrames = BurstFrames[i : j+1]
        BurstChannels = BurstChannels[i : j+1]
    return BurstFrames, BurstChannels

def BurstPlot(bxr, wellID, startTime=0, Duration=0.01):
    """Plot burst events in a selected time interval with color-coded channels.

    Args:
        bxr (BxrFile): file bxr opened from its path
        wellID (str): identifier of the selected well
        startTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.01.

    Returns:
        None (displays matplotlib plot)
    """
    BurstFrames, BurstChannels = Burst2df(bxr, wellID, startTime, Duration)

    if len(BurstFrames) == 0:
        print('No bursts in the time interval ['+str(startTime)+', '+str(startTime+Duration)+']')
    else:
        #starting frame
        startFrame = ConversionTimeToFrames(bxr, startTime)
        #number of frames considered
        NumFrames = ConversionTimeToFrames(bxr, Duration)
        endFrame = startFrame+NumFrames
        SamplingRate = bxr.attrs['SamplingRate']
        if len(BurstFrames)==1:
            if startFrame > BurstFrames[0]:
                BurstFrames[0] = startFrame
            if endFrame < BurstFrames[1]:
                BurstFrames[1] = endFrame
        else:
            if startFrame > BurstFrames[0][0]:
                BurstFrames[0][0] = startFrame
            if endFrame < BurstFrames[len(BurstFrames)-1][1]:
                BurstFrames[len(BurstFrames)-1][1] = endFrame
        BurstChannels_unique = np.unique(BurstChannels)
        BurstTimes_extended = []
        if len(BurstFrames)==1:
            for it in np.arange(len(BurstFrames)):
                f1 = BurstFrames[0]
                f2 = BurstFrames[1]
                BurstTimes_extended.append(np.arange(f1,f2+1)/SamplingRate)
        else: 
            for it in np.arange(len(BurstFrames)):
                f1 = BurstFrames[it, 0]
                f2 = BurstFrames[it, 1]
                BurstTimes_extended.append(np.arange(f1,f2+1)/SamplingRate)
        NumChannels = np.array(bxr[wellID + '/StoredChIdxs']).shape[0]
        count = 0
        data = []
        for it in np.arange(NumChannels):
            aux = np.where(BurstChannels == it)
            if len(aux[0]) > 0:
                tt = np.empty((0,))
                for it_aux in np.arange(len(aux[0])):
                    aux_list = BurstTimes_extended[aux[0][it_aux]]
                    tt = np.concatenate((tt,aux_list))
                data.append(tt)
                count = count+1
            #else:
            #    data.append([])
        colors1 = ['C{}'.format(i) for i in range(count)]
        fig, ax = plt.subplots()
        plt.eventplot(data, colors=colors1)
        plt.title('Burst Plot, Time interval = ['+str(startTime)+', '+str(startTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('(channels)')
        plt.yticks(np.arange(count))
        ax.set_yticklabels(BurstChannels_unique)
        plt.show()

def WaveformsPlot(bxr, wellID, startTime=0, Duration=0.01, chIdx=0):
    """Plot spike waveforms for a specific channel in a selected time interval.

    Args:
        bxr (BxrFile): file bxr opened from its path
        wellID (str): identifier of the selected well
        startTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.01.
        chIdx (int, optional): channel index to plot. Defaults to 0.

    Returns:
        None (displays matplotlib plot)
    """
    #starting frame
    startFrame = ConversionTimeToFrames(bxr, startTime)
    #number of frames considered
    NumFrames = ConversionTimeToFrames(bxr, Duration)   
    # collect the TOCs
    toc = np.array(bxr['TOC'])
    spikeToc = np.array(bxr[wellID + '/SpikeTOC'])

    # collect experiment information
    minDigitalValue = bxr.attrs['MinDigitalValue']
    maxDigitalValue = bxr.attrs['MaxDigitalValue']
    minAnalogValue = bxr.attrs['MinAnalogValue']
    maxAnalogValue = bxr.attrs['MaxAnalogValue']
    dacFactor = (maxAnalogValue - minAnalogValue) / (maxDigitalValue - minDigitalValue)
    offsetValue = minAnalogValue - dacFactor * minDigitalValue
    samplingRate = bxr.attrs['SamplingRate']
    chIdxs = np.array(bxr[wellID + '/StoredChIdxs'])

    # from the given start position and duration (in frames), find the corresponding range of spike positions using the TOC
    tocStartIdx = np.searchsorted(toc[:, 1], startFrame)
    tocEndIdx = min(np.searchsorted(toc[:, 1], startFrame + NumFrames, side='right') + 1, len(toc) - 1)
    spikeStartPosition = spikeToc[tocStartIdx]
    spikeEndPosition = spikeToc[tocEndIdx]

    # collect the required spike data
    spikeDataTimestamps = bxr[wellID + '/SpikeTimes'][spikeStartPosition:spikeEndPosition]
    spikeDataChIdxs = bxr[wellID + '/SpikeChIdxs'][spikeStartPosition:spikeEndPosition]

    spikeSortingPerformed = bxr.__contains__(wellID + '/SpikeUnits')
    if spikeSortingPerformed:
        spikeDataChUnits = bxr[wellID + '/SpikeUnits'][spikeStartPosition:spikeEndPosition]

    waveformLength = bxr[wellID + '/SpikeForms'].attrs['Wavelength']
    spikeDataWaveforms = bxr[wellID + '/SpikeForms'][spikeStartPosition*waveformLength:spikeEndPosition*waveformLength]
    dataLength = spikeEndPosition - spikeStartPosition

    # collect the waveforms for the given time range and channel index
    waveformData = {} if spikeSortingPerformed else []
    ts = []
    for i in range(0, dataLength):
        if spikeDataChIdxs[i] == chIdx and startFrame <= spikeDataTimestamps[i] < startFrame + NumFrames:
            ts.append(spikeDataTimestamps[i])
            if spikeSortingPerformed:
                spikeUnit = spikeDataChUnits[i]
                if spikeUnit not in waveformData.keys():
                    waveformData[spikeUnit] = []
                waveformData[spikeUnit].append(spikeDataWaveforms[i*waveformLength:i*waveformLength+waveformLength])
            else:
                waveformData.append(spikeDataWaveforms[i*waveformLength:i*waveformLength+waveformLength])
    
    # visualize waveforms for the given channel index, if spike sorting was performed,
    # units will be plotted with different colors
    if len(waveformData)==0:
        print('No waveforms for the channel '+str(chIdx)+' in the time interval ['+str(startTime)+', '+str(startTime+Duration)+']')

    else:
        plt.figure()
        x = np.arange(0, waveformLength, 1) / samplingRate

        if spikeSortingPerformed:
            colors = list(mcolors.BASE_COLORS.keys())
            c = 0
            for unit in waveformData:
                for waveform in waveformData[unit]:
                    # convert the waveform to analog
                    y = offsetValue + dacFactor * waveform
                    plt.plot(x, y, color=colors[c])
                c += 1
        else:
            for waveform in waveformData:
                # convert the waveform to analog
                y = offsetValue + dacFactor * waveform
                plt.plot(x, y, color='blue')

        plt.title('Spike waveforms = '+str(len(ts))+', channel = '+ str(chIdx+1)+', '+'Time interval=['+str(startTime)+', '+str(startTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('(uV)')
        plt.legend()
        plt.show()

def FP2df(bxr, wellID, startTime = 0, Duration = 0.05):
    """
    Selected a BXR file and a well, we read when and where we have a FP in a selected time interval

    Args:
        bxr (BXRFile): file bxr opened from its path
        wellID (str): identifier of the selected well
        startTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second) I want to consider. Defaults to 0.05

    Returns:
        FPFrames (array): it contains the frames when FPs occured; 
        FPChannels (array): it contains the channel that measured the FPs
    """    
    startFrame = ConversionTimeToFrames(bxr, startTime)
    endFrame = startFrame + ConversionTimeToFrames(bxr, Duration)
    FPFrames = np.array(bxr[wellID+'/FpTimes'])
    FPChannels = np.array(bxr[wellID+'/FpChIdxs'])
    indexes = np.where((FPFrames>=startFrame)&(FPFrames<=endFrame))[0]
    return FPFrames[indexes], FPChannels[indexes]

def FPFormPlot(bxr, wellID, startTime=0, Duration=0.05, chIdx=0):
    """Plot false positive waveforms for a specific channel in a selected time interval.

    Args:
        bxr (BxrFile): file bxr opened from its path
        wellID (str): identifier of the selected well
        startTime (float, optional): starting time in seconds. Defaults to 0.
        Duration (float, optional): duration of the measurement in seconds. Defaults to 0.05.
        chIdx (int, optional): channel index to plot. Defaults to 0.

    Returns:
        None (displays matplotlib plot)
    """  
    #starting frame
    startFrame = ConversionTimeToFrames(bxr, startTime)
    #number of frames considered
    NumFrames = ConversionTimeToFrames(bxr, Duration)  
    # collect experiment information
    minDigitalValue = bxr.attrs['MinDigitalValue']
    maxDigitalValue = bxr.attrs['MaxDigitalValue']
    minAnalogValue = bxr.attrs['MinAnalogValue']
    maxAnalogValue = bxr.attrs['MaxAnalogValue']
    dacFactor = (maxAnalogValue - minAnalogValue) / (maxDigitalValue - minDigitalValue)
    offsetValue = minAnalogValue - dacFactor * minDigitalValue
    samplingRate = bxr.attrs['SamplingRate']
    FPForms = np.array(bxr[wellID+'/FpForms'])
    FPformLength = bxr[wellID + '/FpForms'].attrs['Wavelength']
    FPFrames, FPChannels = FP2df(bxr, wellID, startTime, Duration)
    if startFrame <= FPFrames[0]:
        first_index = 0
    else:
        f_i = np.where(FPFrames>startFrame)[0]
        first_index = f_i[0]
    if startFrame+NumFrames>=FPFrames[len(FPFrames)-1]:
        second_index = len(FPFrames)-1
        index = np.where(FPChannels[first_index:second_index+1]==chIdx)[0]
    elif startFrame+NumFrames<=FPFrames[0]:
        second_index = 0
        index = []
    else: 
        s_i = np.where(FPFrames<=startFrame+NumFrames)[0]
        second_index = s_i[len(s_i)-1]
        index = np.where(FPChannels[first_index:second_index+1]==chIdx)[0]
    if len(index)>0:    
        plt.figure()
        x = np.arange(0, FPformLength, 1)/samplingRate
        colors = list(mcolors.XKCD_COLORS.keys())
        c = 0
        for i in index:
            y = offsetValue + dacFactor * FPForms[i:i + FPformLength]
            plt.plot(x, y, color=colors[c])
            c += 1
        
        plt.title('FPforms = ' +str(len(index))+ ', channel = '+ str(chIdx+1)+', Time interval = ['+str(startTime)+', '+str(startTime+Duration)+']')
        plt.xlabel('(sec)')
        plt.ylabel('uV')
        plt.legend()
        plt.show()
    else: 
        print('No FP for the channel '+ str(chIdx+1)+' in the time interval ['+str(startTime)+', '+str(startTime+Duration)+']')

def SpikesDataset(brw, bxr, wellID, Downsampling_Frequency, StartTime=0, Duration=0.05, ch=-10):
    """Generate a dataset of time windows centered on spike events from a selected channel.

    Args:
        brw (BrwFile): file with raw data
        bxr (BxrFile): file with analyzed data
        wellID (str): well identifier for the study
        Downsampling_Frequency (float): sampling frequency for downsampling
        StartTime (float, optional): start time in seconds. Defaults to 0.
        Duration (float, optional): duration of analysis window in seconds. Defaults to 0.05.
        ch (int, optional): channel ID. If negative, selects channel with highest spike count. Defaults to -10.

    Returns:
        np.ndarray: dataset array where each row is a 40-frame window (based on sampling frequency) 
                    centered on a spike from the selected channel. Shape is [num_spikes, window_length].
    """           
    #reading
    SamplingRate = float(brw.attrs['SamplingRate'])
    SpikeFrames, SpikeChannels = Spikes2df(bxr, wellID, StartTime, Duration)
    num_spikes = len(SpikeFrames)

    Spikes_for_channels = np.zeros(num_spikes)
    for i in range(num_spikes):
        ch = SpikeChannels[i]
        Spikes_for_channels[ch] = Spikes_for_channels[ch]+1
    max = np.max(Spikes_for_channels)
    ch_max = np.where(Spikes_for_channels == max)[0]
    if len(ch_max)==1:
        ch_max = ch_max
    else:
        ch_max = ch_max[0]

    if ch < 0:
        ch_max = ch_max
    else: 
        ch_max = ch

    t=time.time()
    DataFull_1m, frames_index_1m = brw_f.ReadingRawData(brw, wellID, SamplingRate, 0, 60)
    print(time.time()-t)
    DataFull_2m, frames_index_2m = brw_f.ReadingRawData(brw, wellID, SamplingRate, 60, 60)
    print(time.time()-t)
    DataFull_3m, frames_index_3m = brw_f.ReadingRawData(brw, wellID, SamplingRate, 120, Duration-120)
    print(time.time()-t)
    

    f_1m = len(frames_index_1m)
    f_2m = len(frames_index_2m)
    f_3m = len(frames_index_3m)
    NumFrames = f_1m + f_2m + f_3m
    # DataChannel = np.zeros((NumFrames,1))
    DataChannel = np.zeros(NumFrames)
    DataChannel[0:f_1m] = DataFull_1m[:,ch_max]
    DataChannel[f_1m:f_1m+f_2m] = DataFull_2m[:,ch_max]
    DataChannel[f_2m+f_1m:NumFrames] = DataFull_3m[:,ch_max]

    idx_ch_max = np.where(SpikeChannels==ch_max)[0]
    spikes_ch_max = SpikeFrames[idx_ch_max]

    data = []
    frames = []

    step = round(SamplingRate/Downsampling_Frequency)

    for i in range(len(spikes_ch_max)):
        curr_spike_window = DataChannel[spikes_ch_max[i]-20*step:spikes_ch_max[i]+20*step+1]
        idx_NB = np.where(curr_spike_window==-8000)[0]
        if len(idx_NB)==0:
            data.append(curr_spike_window)
            frames.append(spikes_ch_max[i])

    l = len(data)
    w = len(data[0])
    x = np.arange(0,w,step)
    data = np.array(data)
    # dataset = np.zeros((l,w,1))
    dataset = np.zeros((l,w))
    for i in range(l):
        dataset[i] = data[i]
    
    curr_frame = frames[0]
    indexes = []
    indexes.append(0)
    for i in range(l-1):
        if frames[i+1]>=curr_frame+40*step:
            indexes.append(i+1)
            curr_frame = frames[i+1]

    dataset = dataset[indexes]
    frames = np.array(frames)
    frames = frames[indexes]
    spikes_pos = []
    spikes_neg = []
    for i in range(len(frames)):
        if DataChannel[frames[i]]>0:
            spikes_pos.append(i)
        else:
            spikes_neg.append(i)

    dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1]))
    dataset = dataset[:, x]

    return dataset





