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
from . import bxr_functions
import time
from scipy.signal import find_peaks, butter, filtfilt, wiener, iirnotch
from statistics import median
import plotly.express as px
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns

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
        filename (str): name of the file and its extension .brw 
        wellID (str): identifier of the selected well 

    Returns:
        brw (h5py): the brw data
    """    
    brw = h5py.File(filename)

    toc = np.array(brw['TOC'])
    NumFrames = toc[toc.shape[0]-1,1] 
    SamplingRate= brw.attrs['SamplingRate']
    NumChannels = np.array(brw[wellID + '/StoredChIdxs']).shape[0]
    Duration = NumFrames/SamplingRate

    print('--- File: ' + filename + ' ---')
    print('Number of Channels: ' + str(NumChannels))
    print('File Duration: ' + str(Duration))
    print('Total Number of Frames: ' + str(NumFrames))
    print('Sampling Frequency: ' + str(SamplingRate) + ' Hz')
    print('---')

    return brw


def Seconds2Frames(brw, Time):
    """
    Conversion from time in seconds and time in frames

    Args:
        brw (BrwFile): file brw opened from its path
        Time (float): time in seconds

    Returns:
        Frame (int): time in frames
    """    
   
    SamplingRate = brw.attrs['SamplingRate']
    Frame = int(Time * SamplingRate)

    return Frame


def DecodeEventBasedRawData(brw, data, wellID, startTime=0, Duration=0.05):
    """ FROM 3BRAIN 

    Args:
        brw (BrwFile): file brw opened from its path
        data (dictionary): the keys are the recorded channel indexes StoredChIdxs and the values an array initialized with numFrames zeros for each key
        wellID (str): identifier of the selected well
        Downsampling_Frequency (float): chosen sampling frequency 
        startTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05

    Returns:
        data (list): The returned data list contains digital samples that can be converted into analog values
    """   
    startFrame = Seconds2Frames(brw, startTime)
    endFrame = Seconds2Frames(brw, startTime+Duration)
    # collect the TOCs
    toc = np.array(brw['TOC']) #dà errore con i dati vecchi (ho provato DataSet_02)
    if endFrame < startFrame:
        endFrame = toc[toc.shape[0]-1,1]
    
    eventsToc = np.array(brw[wellID + '/EventsBasedSparseRawTOC'])
    # from the given start position and duration in frames, localize the corresponding event positions
    # using the TOC
    tocStartIdx = np.searchsorted(toc[:, 1], startFrame)
    tocEndIdx = min(np.searchsorted(toc[:, 1], endFrame, side='right')+ 1, len(toc) - 1)
    eventsStartPosition = eventsToc[tocStartIdx]
    eventsEndPosition = eventsToc[tocEndIdx]
    # decode all data for the given well ID and time interval
    binaryData = brw[wellID + '/EventsBasedSparseRaw'][eventsStartPosition:eventsEndPosition]
    binaryDataLength = len(binaryData)
    pos = 0
    while pos < binaryDataLength:
        chIdx = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        chDataLength = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        chDataPos = pos
        while pos < chDataPos + chDataLength:
            fromInclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
            pos += 8
            toExclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
            pos += 8
            rangeDataPos = pos
            for j in range(fromInclusive, toExclusive):
                if j >= endFrame:
                    break
                if j >= startFrame:
                    data[chIdx][j - startFrame] = int.from_bytes(binaryData[rangeDataPos:rangeDataPos + 2], byteorder='little', signed=True)

                rangeDataPos += 2
            pos += (toExclusive - fromInclusive) * 2

    return data 

def ReadingRawData(brw, wellID, DownsamplingFrequency, StartTime = 0, Duration = 0.05): 
    """

    Args:
        brw (BrwFile): file brw opened from its path
        wellID (str): identifier of the selected well
        Downsampling_Frequency (float): chosen sampling frequency 
        startTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05

    Returns:
        aux_data (array): array that contains the signals measured from StartFrame to EndFrame
        frames2save (array): array that contains the frames relative to measurements in aux_data
    """
    startFrame = Seconds2Frames(brw, StartTime)
    endFrame = Seconds2Frames(brw, StartTime+Duration)
    # collect experiment information
    minDigitalValue = brw.attrs['MinDigitalValue']
    maxDigitalValue = brw.attrs['MaxDigitalValue']
    minAnalogValue = brw.attrs['MinAnalogValue']
    maxAnalogValue = brw.attrs['MaxAnalogValue']
    dacFactor = (maxAnalogValue - minAnalogValue) / (maxDigitalValue - minDigitalValue)
    offsetValue = minAnalogValue - dacFactor * minDigitalValue

    toc = np.array(brw['TOC'])
    if endFrame < startFrame:
            endFrame = toc[toc.shape[0]-1,1]

    chIdxs = np.array(brw[wellID + '/StoredChIdxs'])
    chIdxs.sort()#
    n_ch = len(chIdxs)#
    NumChannels = chIdxs.shape[0]
    SamplingRate= brw.attrs['SamplingRate']
    if 'EventsBasedSparseRawTOC' in brw[wellID]:
        data_dict = {}
        for chIdx in chIdxs:
            data_dict[chIdx] = np.zeros(endFrame-startFrame, dtype=np.int16) 
        data_dict = DecodeEventBasedRawData(brw, data_dict, wellID, StartTime, Duration)

        data = np.zeros((endFrame-startFrame, n_ch))
        for d in range(n_ch):
            data[:, d] = np.array(data_dict[chIdxs[d]], dtype=float)
        '''
        for chIdx in chIdxs:
            index = np.where(chIdxs == chIdx)[0][0]
            data[:,index] = np.fromiter(data_dict[chIdxs[index]], float)
        '''


    elif 'Raw' in brw[wellID]:
        aux_data = brw[wellID + '/Raw'] 
        aux_data = aux_data[startFrame*NumChannels:endFrame*NumChannels]
        data = np.reshape(aux_data, (endFrame-startFrame, NumChannels))

    elif 'WaveletBasedEncodedRaw' in brw[wellID]: 
        coefsTotalLength = len(brw[wellID + '/WaveletBasedEncodedRaw'])
        compressionLevel = brw[wellID + '/WaveletBasedEncodedRaw'].attrs['CompressionLevel']
        framesChunkLength = brw[wellID + '/WaveletBasedEncodedRaw'].attrs['CompressionLevel']
        coefsChunkLength = math.ceil(framesChunkLength/pow(2, compressionLevel))*2
        for chIdx in chIdxs:
            t = time.time()
            data = []
            coefsPosition = chIdx * coefsChunkLength
            while coefsPosition < coefsTotalLength:
                coefs = brw[wellID + '/WaveletBasedEncodedRaw'][coefsPosition:coefsPosition+coefsChunkLength]
                length = int(len(coefs)/2)
                frames = pywt.idwt(coefs[:length], coefs[length:], 'sym7', 'periodization') 
                length *= 2
                for i in range(1, compressionLevel):
                    frames = pywt.idwt(frames[:length], None, 'sym7', 'periodization')
                    length *= 2
                data.extend(frames)
                coefsPosition += coefsChunkLength * NumChannels
            print(time.time()-t) 
            print("un canale")   
        brw.close()


    step = int(SamplingRate/DownsamplingFrequency)

    frames2save = np.arange(0, endFrame-startFrame, step)
    aux_data = np.empty((len(frames2save), data.shape[1]))
    for f in np.arange(len(frames2save)):
        if int(frames2save[f]) == endFrame-startFrame:
            aux_data[f, :] = data[int(frames2save[f])-1, :]
        else:
            aux_data[f, :] = data[int(frames2save[f]),:]

    frames2save = np.array(frames2save, dtype = int)

    aux_data = offsetValue + dacFactor * aux_data

    return aux_data, frames2save+startFrame

def ReadingSingleChannel(brw, wellID, DownsamplingFrequency, row, col, StartTime = 0, Duration = 0.05):#to modify 
    """
    Selected a BRW file, a well and a channel in the well, 
    this function reads the activity signal of the channel during the experiment

    Args:
        brw (BrwFile): file brw opened from its path
        wellID (str): identifier of the selected well
        DownsamplingFrequency (float): chosen sampling frequency 
        row (int): number from 0 to the maximum number of channel rows 
        col (int): number from 0 to the maximum number of channel columns
        StartTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05

    Returns:
        aux_data (array): array that contains the signals measured in the selected channel
        frames2save (array): array that contains the frames relative to measurements in aux_data
    """

    data, frames2save = ReadingRawData(brw, wellID, DownsamplingFrequency, StartTime, Duration)
    chIdxs = np.array(brw[wellID + '/StoredChIdxs'])
    dim_1 = int(np.sqrt(chIdxs.shape[0]))
    aux_data = data[:,row*dim_1+col]
    return aux_data, frames2save

def LoopReading(ch, NumChannels, Duration, TotDuration, brw, wellID, SamplingRate, StartTime, outputPath):
    """Read one channel over consecutive windows and save it as .npy.

    Args:
        ch (int): channel index.
        NumChannels (int): number of channels per row (e.g. 64).
        Duration (float): duration of each read chunk in seconds.
        TotDuration (float): total duration to read in seconds.
        brw (BrwFile): opened BRW file handle.
        wellID (str): selected well identifier.
        SamplingRate (float): target sampling frequency used by ReadingSingleChannel.
        StartTime (float): initial time in seconds.
        outputPath (str): folder where channel .npy file is saved.
    """
    row = np.int16(np.floor(ch / NumChannels))
    col = ch - NumChannels * row
    frames_index = np.array([])
    data_full = np.array([])
    count = 0
    while (count + 1) * Duration < TotDuration:
        data_full_chunk, frames_index_chunk = ReadingSingleChannel(
            brw,
            wellID,
            SamplingRate,
            row,
            col,
            StartTime + count * Duration,
            Duration,
        )
        count = count + 1
        data_full = np.concatenate([data_full, data_full_chunk.copy()])
        frames_index = np.concatenate([frames_index, frames_index_chunk.copy()])
    np.save(outputPath + '/ch' + str(ch) + '.npy', data_full)

    return frames_index

def PlotRawData(brw, wellID, title, Downsampling_Frequency, row, col, startTime=0, Duration=0.05): 
    """
    Selected a BRW file, a well and a channel in the well, 
    this function prints a graphic of the channel activity signal

    Args:
        brw (BrwFile): file brw opened from its path
        wellID (str): identifier of the selected well
        Downsampling_Frequency (float): chosen sampling frequency 
        row (int): number from 0 to 63 that selects the row of the channel in the well
        col (int): number from 0 to 63 that selects the column of the channel in the well
        startTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05
    """    
    startFrame = Seconds2Frames(brw, startTime)
    endFrame = Seconds2Frames(brw, startTime+Duration)
    SamplingRate = brw.attrs['SamplingRate']
    y, frames2save = ReadingSingleChannel(brw, wellID, Downsampling_Frequency, row, col, startTime, Duration)
    x = frames2save/SamplingRate
    y = np.transpose(y)

    plt.figure()
    plt.plot(x, y, color="blue")
    plt.title('Raw Signal of the channel '+ str(row*64 + col) +', time interval = ['+str(round(startFrame/SamplingRate*100)/100)+', '+str(round(endFrame/SamplingRate*100)/100)+']')
    plt.xlabel('(sec)')
    plt.ylabel('(uV)')
    plt.savefig(title+".png")
    plt.show()


def SaveChannel(args_tuple):
    """Append channel data to an existing .npy file and save it.

    Args:
        args_tuple (tuple): (ch, data_ch, basePath, prefix)
            - ch (int): channel index.
            - data_ch (array-like): new samples to append.
            - basePath (str): folder where channel files are stored.
            - prefix (str): filename prefix (e.g. 'channel_' or 'ch').
    """
    ch, data_ch, basePath, prefix = args_tuple
    channelFile = basePath + '/' + prefix + str(ch) + '.npy'
    data_ch_prev = np.load(channelFile)
    data_ch = np.array(list(data_ch) + list(data_ch_prev))
    np.save(channelFile, data_ch)

def SingleChannelFramesWithPeaks(brw, wellID, Downsampling_Frequency, row, col, startTime=0, Duration = 0.05, threshold=0):
    """
    This function returns the frames for the channel define by row and colum
    where we have a peak larger than a threshold defined by the user

    Args:
        brw (BrwFile): file brw opened from its path
        wellID (str): identifier of the selected well
        Downsampling_Frequency (float): chosen sampling frequency
        row (int): number from 0 to 63 that selects the row of the channel in the well
        col (int): number from 0 to 63 that selects the column of the channel in the well
        startTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05
        threshold (float, optional): the threshold on the level of activity. Defaults to 0

    Returns:
       frames_with_peaks (array): array that contains the frames of the peaks 
    """    
    y, frames2save = ReadingSingleChannel(brw, wellID, Downsampling_Frequency, row, col, startTime, Duration)
    y = np.transpose(y)
    peaks = scipy.signal.find_peaks(y, threshold=threshold)
    frames_with_peaks = frames2save[peaks[0]]
    return frames_with_peaks 

def FramesWithPeaks(brw, wellID, Downsampling_Frequency, startTime = 0, Duration = 0.05, Percentage = 0, threshold=0):
    """
    This function returns the frames where we have peaks
    larger or lower than a threshold defined by the user

    Args:
        brw (BrwFile): file brw opened from its path
        wellID (str): identifier of the selected well
        Downsampling_Frequency (float): chosen sampling frequency
        startTime (c, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05
        Percentage (int, optional): Percentage of the channel with peaks. Defaults to 0
        threshold (float, optional): the threshold on the level of activity. Defaults to 0

    Returns:
        MatrixPeaks: (array): array that contains the frames of the peaks for all the channels
        frames_underperc (list): it contains the frames with a number of peaks smaller than Percentage*NumChannels
        frames_overperc (list): it contains the frames with a number of peaks larger than Percentage*NumChannels

    """   
    data, frames2save = ReadingRawData(brw, wellID, Downsampling_Frequency, startTime, Duration)
    numChannels = data.shape[1]
    numFrames2Save = len(frames2save)
    MatrixPeaks = np.zeros((numFrames2Save, numChannels))

    for ch in range(numChannels):
        indexPeaks = scipy.signal.find_peaks(data[:,ch], threshold=threshold)
        MatrixPeaks[indexPeaks[0], ch] = 1

    numPeaks = np.sum(MatrixPeaks, axis = 1)
    frames_overperc = []
    frames_underperc = []
    PC = Percentage*numChannels
    for t in range(numFrames2Save):
        if numPeaks[t]>=PC:
            frames_overperc.append(numPeaks[t])
        else:
            frames_underperc.append(numPeaks[t])
    
    return  MatrixPeaks, frames_underperc, frames_overperc

def BRW2df(brw, wellID, Downsampling_Frequency, startTime = 0, Duration = 0.05): 

    """
    Selected a BrwFile and a well, this function creates 2 dataframe: 
    - one with the coordinates of the channels and 
    - one with the evolution over time of the activity maps of the well

    Args:
        brw (BrwFile): file brw opened from its path
        wellID (str): identifier of the selected well
        Downsampling_Frequency (float): chosen sampling frequency
        startTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05

    Returns:
        DataFrame, DataFrame:
        -df_XY is a pandas dataframe where we saved the couples (X,Y) that indicate the coordinates of the channels (we read channels row by row)
        -df_AL is a pandas data frame where we saved frames and their respective activity maps vectorized like we read channels  
    """       
    dim_1 = int(np.sqrt(np.array(brw[wellID + '/StoredChIdxs']).shape[0]))
    dim_2 = dim_1
    data, frames2save = ReadingRawData(brw, wellID, Downsampling_Frequency, startTime, Duration)

    lista_AL = []
    for it in np.arange(data.shape[0]):
        aux = data[it,:]
        tupla_AL = (int(frames2save[it]), aux)
        lista_AL.append(tupla_AL)

    lista_XY  = []
    for it in np.arange(1,dim_1+1):
        tupla_XY = (it, np.arange(1,dim_2+1))
        lista_XY.append(tupla_XY)

    df_XY = pd.DataFrame(lista_XY, columns=["X", "Y"])
    df_AL = pd.DataFrame(lista_AL, columns=["Frame", "Activity"])
    return df_XY, df_AL

def SpikesActivityLevel(brw, bxr, wellID, startTime = 0, Duration = 0.05):
    """
    This function creates a matrix (number of frames saved x channels) where the entry ij is not zero if and only if
    the channel j has a spike at frame i; the entry is equal to the activity level of the channel j at frame i

    Args:
        brw (BrwFile): BRW file
        bxr (BXRFile): BRW file
        wellID (str): identifier of the selected well
        startTime (float, optional): starting time in seconds. Defaults to 0
        Duration (float, optional): duration of the measurement (in second). Defaults to 0.05

    Returns:
        spikes_AL (array): matrix where the entry ij is not zero if and only if the channel j has a spike at frame i 
    """    
    SamplingRate= brw.attrs['SamplingRate']
    startFrame = Seconds2Frames(brw, startTime) 
    SpikeFrames, SpikeChannels = bxr_functions.Spikes2df(bxr, wellID, startTime, Duration)
    data, frames2save = ReadingRawData(brw, wellID, SamplingRate, startTime, Duration)
    spikes_AL = np.zeros((data.shape[0],data.shape[1]))
    for i in range(len(SpikeFrames)):
        print('Spike at frame '+str(SpikeFrames[i])+', channel number '+str(SpikeChannels[i]+1))
        spikes_AL[SpikeFrames[i]-startFrame-1][SpikeChannels[i]] = data[SpikeFrames[i]-startFrame-1][SpikeChannels[i]] 
    return spikes_AL  

def BandpassFilter(data, lowcut, highcut, SamplingRate):
    """Band pass filter (lowercase variant)
    
    Args:
        data (array): signals to be filtered
        lowcut (float): lower limit of the band frequency
        highcut (float): upper limit of the band frequency
        SamplingRate (float): signal sampling rate

    Returns:
        array: the filtered signal
    """
    b,a = butter(3, [lowcut/(0.5*SamplingRate), highcut/(0.5*SamplingRate)], btype = 'band' )
    filtered = filtfilt(b, a, data)
    return filtered

def HighpassFilter(data, cut, SamplingRate, order=3):
    """High pass filter (lowercase variant)
    
    Args:
        data (array): signals to be filtered
        cut (float): frequency to remove from the signal
        SamplingRate (float): signal sampling rate
        order (int, optional): the order of the filter. Defaults to 3.

    Returns:
        array: the filtered signal
    """
    b, a = butter(order, cut / (0.5 * SamplingRate), btype='high')
    filtered = filtfilt(b, a, data)
    return filtered

def NotchFilterAlt(data, cut, SamplingRate):
    """Notch filter (alternate naming)
    
    Args:
        data (array): signals to be filtered
        cut (float): frequency to remove from the signal
        SamplingRate (float): signal sampling rate

    Returns:
        array: the filtered signal
    """
    b,a = iirnotch(cut, 30.0, SamplingRate)
    filtered = filtfilt(b, a, data)
    return filtered

def LowpassFilterAlt(data, cut, SamplingRate):
    """Low pass filter (lowercase variant)
    
    Args:
        data (array): signals to be filtered
        cut (float): frequency to remove from the signal
        SamplingRate (float): signal sampling rate

    Returns:
        array: the filtered signal
    """
    b, a = butter(3, cut / (0.5 * SamplingRate), btype='low')
    filtered = filtfilt(b, a, data)
    return filtered

def CommonAverageReferenceAlt(data):
    """Common Average Reference (alternate naming)
    
    Args:
        data (array): signals to be transformed

    Returns:
        array: the transformed signal
    """
    mediana = np.median(data, 1)
    data = (data.T - mediana).T
    mu = np.mean(data,0)
    data = data - mu
    return data

def WienerFilterAlt(data):
    """Wiener filter (alternate naming)
    
    Args:
        data (array): signals to be filtered

    Returns:
        array: the filtered signal
    """
    data = wiener(data)
    return data

def PercentileFilterAlt(data, percentile):
    """Percentile filter (lowercase variant)
    
    Args:
        data (array): signals to be filtered
        percentile (float): the percentile threshold to remove from the data

    Returns:
        array: the filtered signal
    """
    spettro = np.fft.fft(data)
    magnitude = np.abs(spettro)
    threshold = np.percentile(magnitude, percentile)
    spettro[magnitude < threshold] = 0
    filtered = np.fft.ifft(spettro)
    return filtered


'''Metrics discussed with Mark and Maurits'''

def SpikesMetric(SamplingRate, NumFrames, spikes_N, spikes_P, dataset_N, dataset_P, threshold=0.25):
    
    
    
    n_chs = len(spikes_N) 
    Spikes_count = np.zeros(n_chs)
    ISI_map = np.zeros(n_chs)
    ISI_variance_map = np.zeros(n_chs) 
    peak_to_peak_map = np.zeros(n_chs)
    peak_to_peak_std = np.zeros(n_chs)
    
    spikes_frames_well = [] 
    peak_to_peak_well = [] 
    chs = np.arange(n_chs)
    
    for ch in range(n_chs):
        spikes_frames_N = spikes_N[ch] 
        spikes_frames_P = spikes_P[ch] 
        spikes_frames = np.array(sorted(list(set(list(spikes_frames_N))|set(list(spikes_frames_P)))))
        dataset_spikes = np.zeros((len(spikes_frames), 41))
        peak_to_peak_ch = np.zeros(len(spikes_frames))
        if len(spikes_frames)>0:
            dataset_spikes[0:len(spikes_frames_N)]=dataset_N[ch]
            dataset_spikes[len(spikes_frames_N):len(spikes_frames)]=dataset_P[ch]
            for k in range(len(spikes_frames)): 
                peak_to_peak_ch[k] = np.max(dataset_spikes[k])-np.min((dataset_spikes[k]))
            peak_to_peak_well.append(peak_to_peak_ch)

            spikes_frames_well.append(spikes_frames)
            peak_to_peak_map[ch] = np.mean(peak_to_peak_ch)
            peak_to_peak_std[ch] = np.std(peak_to_peak_ch)/peak_to_peak_map[ch]
            Spikes_count[ch]=len(spikes_frames_P)+len(spikes_frames_N)
            if len(spikes_frames)>1:
                ISI_ch = np.diff(spikes_frames)/SamplingRate*1000
                ISI_ch_variance = np.std(ISI_ch)
                ISI_map[ch] = np.mean(ISI_ch)
                ISI_variance_map [ch] = ISI_ch_variance/ISI_map[ch]
        else: 
            spikes_frames_well.append(np.array([]))
            peak_to_peak_well.append(np.array([]))

    Spikes_rate = Spikes_count/NumFrames*SamplingRate

    Active_electrodes_maps = Spikes_rate.copy()
    idxs = np.where(Active_electrodes_maps>=threshold)
    Active_electrodes_ID = idxs[0]
    Non_Active_electrodes_ID = list(np.arange(n_chs))
    Active_electrodes_number = len(Active_electrodes_ID)

    spikes_frames_AE = set()
    peak_to_peak_AE = [] 
    for ch in Active_electrodes_ID:
        Non_Active_electrodes_ID.remove(ch)
        spikes_frames_AE = spikes_frames_AE|set(spikes_frames_well[ch])
        for p in peak_to_peak_well[ch]: 
            peak_to_peak_AE.append(p)
    Spikes_count[Non_Active_electrodes_ID]=0
    peak_to_peak_map[Non_Active_electrodes_ID]=0 
    peak_to_peak_std[Non_Active_electrodes_ID]=0 
    Spikes_rate[Non_Active_electrodes_ID]=0  
    ISI_map[Non_Active_electrodes_ID]=0 
    ISI_variance_map[Non_Active_electrodes_ID]=0  

    spikes_frames_AE = np.array(sorted(spikes_frames_AE))

    ISI_well = np.diff(spikes_frames_AE)/SamplingRate*1000

    ISI_well_average = np.mean(ISI_well)

    ISI_variance_well = np.std(ISI_well)/ISI_well_average

    n_spikes_well = np.sum(Spikes_count)

    spikes_well_rate = n_spikes_well/NumFrames*SamplingRate

    peak_to_peak_average_well = np.mean(peak_to_peak_map)
    peak_to_peak_std_well = np.std(np.array(peak_to_peak_AE))/peak_to_peak_average_well

    return SamplingRate, NumFrames, spikes_N, spikes_P, spikes_frames_AE, Spikes_count, Spikes_rate, ISI_map, ISI_variance_map, ISI_well_average, ISI_variance_well, n_spikes_well, spikes_well_rate, Active_electrodes_ID, Active_electrodes_number, peak_to_peak_map, peak_to_peak_std, peak_to_peak_average_well, peak_to_peak_std_well
    
def BurstsMetric(SamplingRate, NumFrames, spikes_N, spikes_P, dataset_N, dataset_P, threshold, n_min_spikes=5, ISI_max_seconds=0.1):
    
    SamplingRate, NumFrames, spikes_N, spikes_P, spikes_frames_AE, Spikes_count, Spikes_rate, ISI_map, ISI_variance_map, ISI_well_average, ISI_variance_well, n_spikes_well, spikes_well_rate, Active_electrodes_ID, Active_electrodes_number, peak_to_peak_map, peak_to_peak_std, peak_to_peak_average_well, peak_to_peak_std_well = Spikes_metric(SamplingRate, NumFrames, spikes_N, spikes_P, dataset_N, dataset_P, threshold) 

    n_chs = len(spikes_N)
    dim = int(np.sqrt(Spikes_count.shape[0])) 
    bursts = [[] for _ in range(n_chs)]
    IBI = [[] for _ in range(n_chs)]
    IBI_average = [] 
    IBI_std = [] 
    bursts_duration = [[] for _ in range(n_chs)]
    bursts_n_spikes = [[] for _ in range(n_chs)]
    bursts_duration_average = []
    bursts_n_spikes_average = [] 
    bursts_ISI = [[] for _ in range(n_chs)]
    bursts_ISI_average = [] 
    n_bursts = []
    ISI_max_frames = int(SamplingRate*ISI_max_seconds)
    t=time.time()
    for ch in range(n_chs):
        if ch in Active_electrodes_ID:
            spikes_ch = np.array(sorted(set(spikes_N[ch])|set(spikes_P[ch])))
            if len(spikes_ch)>=n_min_spikes:
                i = 0  
                while i <=len(spikes_ch)-n_min_spikes:
                    idx_burst_start = i
                    burst_start = spikes_ch[i]
                    if spikes_ch[i+n_min_spikes-1] <= burst_start+int(SamplingRate*ISI_max_seconds*(n_min_spikes-1)):
                        ISI_bursts = np.diff(spikes_ch[i:i+n_min_spikes])
                        if np.max(ISI_bursts)<=ISI_max_frames:
                            i = i+n_min_spikes-1
                            while i<len(spikes_ch)-1 and spikes_ch[i+1] <= spikes_ch[i] + ISI_max_frames:
                                i = i+1
                            idx_burst_end = i
                            burst_end = spikes_ch[i]
                            bursts[ch].append(spikes_ch[idx_burst_start:idx_burst_end+1])
                            bursts_n_spikes[ch].append(len(spikes_ch[idx_burst_start:idx_burst_end+1])) 
                            bursts_duration[ch].append((spikes_ch[idx_burst_end]-spikes_ch[idx_burst_start])/SamplingRate*1000)
                            bursts_ISI[ch].append(((spikes_ch[idx_burst_end]-spikes_ch[idx_burst_start])/(len(spikes_ch[idx_burst_start:idx_burst_end+1])-1))/SamplingRate*1000)  
                            i=i+1
                        else: 
                            i=i+1
                    else:
                        i=i+1
            n_bursts.append(len(bursts[ch])) 
            if n_bursts[ch]>0: 
                bursts_duration_average.append(np.mean(np.array(bursts_duration[ch])))
                bursts_n_spikes_average.append(np.mean(np.array(bursts_n_spikes[ch])))
                bursts_ISI_average.append(np.mean(np.array(bursts_ISI[ch])))
            else:
                bursts_duration_average.append(0)
                bursts_n_spikes_average.append(0)
                bursts_ISI_average.append(0)
            if n_bursts[ch]>1: 
                for j in range(n_bursts[ch]-1):
                    IBI[ch].append((bursts[ch][j+1][0]-bursts[ch][j][0])/SamplingRate*1000) 
                IBI_average.append(np.mean(np.array(IBI[ch])))
                IBI_std.append(np.std(np.array(IBI[ch]))/IBI_average[ch])
            else:
                IBI_average.append(0)
                IBI_std.append(0)
        else:   
            n_bursts.append(len(bursts[ch]))
            bursts_duration_average.append(0)
            bursts_n_spikes_average.append(0)
            IBI_average.append(0)
            IBI_std.append(0)
            bursts_ISI_average.append(0)

    n_bursts = np.array(n_bursts)
    bursts_rate = n_bursts/NumFrames*SamplingRate*60
    bursts_n_spikes_average = np.array(bursts_n_spikes_average)
    bursts_spikes_percentage = np.zeros(n_chs) 
    for ch in range(n_chs):
        if Spikes_count[ch]>0 and n_bursts[ch]>0:  
            bursts_spikes_percentage[ch]  = n_bursts[ch]*bursts_n_spikes_average[ch]/n_spikes_well*100  #vedere bene questa definizione
    bursts_ISI_average = np.array(bursts_ISI_average)
    bursts_duration_average = np.array(bursts_duration_average)
    IBI_average = np.array(IBI_average) 
    IBI_std = np.array(IBI_std)

    bursts_well = []
    bursts_duration_well = [] 
    bursts_ISI_well = [] 
    bursts_n_spikes_well = [] 
    i = 0  
    while i <=len(spikes_frames_AE)-n_min_spikes:
        idx_burst_start = i
        burst_start = spikes_frames_AE[i]
        if spikes_frames_AE[i+n_min_spikes-1] <= burst_start+ISI_max_frames*(n_min_spikes-1):
            ISI_bursts = np.diff(spikes_frames_AE[i:i+n_min_spikes])
            if np.max(ISI_bursts)<=ISI_max_frames:
                i = i+n_min_spikes-1
                while i<len(spikes_frames_AE)-1 and spikes_frames_AE[i+1] <= spikes_frames_AE[i] + ISI_max_frames:
                    i = i+1
                idx_burst_end = i
                burst_end = spikes_frames_AE[i]
                bursts_well.append(spikes_frames_AE[idx_burst_start:idx_burst_end+1])
                bursts_n_spikes_well.append(len(spikes_frames_AE[idx_burst_start:idx_burst_end+1])) 
                bursts_duration_well.append((spikes_frames_AE[idx_burst_end]-spikes_frames_AE[idx_burst_start])/SamplingRate*1000) 
                bursts_ISI_well.append(((spikes_frames_AE[idx_burst_end]-spikes_frames_AE[idx_burst_start])/(len(spikes_frames_AE[idx_burst_start:idx_burst_end+1])-1))/SamplingRate*1000)
                i=i+1
            else: 
                i=i+1
        else:
            i=i+1 
    n_bursts_well = len(bursts_well) 
    bursts_rate_well = n_bursts_well/NumFrames*SamplingRate*60
    bursts_n_spikes_well = np.array(bursts_n_spikes_well)
    bursts_duration_well = np.array(bursts_duration_well)
    bursts_ISI_well = np.array(bursts_ISI_well)
    bursts_n_spikes_well_average = np.mean(bursts_n_spikes_well)
    bursts_n_spikes_well_std = np.std(bursts_n_spikes_well)/bursts_n_spikes_well_average
    bursts_duration_well_average = np.mean(bursts_duration_well)
    bursts_duration_well_std = np.std(bursts_duration_well)/bursts_duration_well_average
    bursts_ISI_well_average = np.mean(bursts_ISI_well)
    bursts_ISI_well_std = np.std(bursts_ISI_well)/bursts_ISI_well_average
    bursts_spikes_percentage_well = bursts_n_spikes_well_average*n_bursts_well/n_spikes_well*100

    IBI_well = [] 
    if n_bursts_well>1:
        for j in range(n_bursts_well-1):
            IBI_well.append(bursts_well[j+1][0]-bursts_well[j][0])
    IBI_well = np.array(IBI_well)/SamplingRate*1000
    IBI_well_average = np.mean(IBI_well)
    IBI_well_std = np.std(IBI_well)/IBI_well_average

    return Active_electrodes_ID, bursts, n_bursts, bursts_rate, bursts_n_spikes, bursts_spikes_percentage, bursts_ISI, bursts_ISI_average, bursts_n_spikes_average, bursts_duration, bursts_duration_average, IBI, IBI_average, IBI_std, bursts_well, n_bursts_well, bursts_rate_well, bursts_duration_well_average, bursts_duration_well_std, IBI_well, IBI_well_average, IBI_well_std, bursts_n_spikes_well, bursts_n_spikes_well_average, bursts_n_spikes_well_std, bursts_ISI_well, bursts_ISI_well_average, bursts_ISI_well_std, bursts_spikes_percentage_well, SamplingRate, NumFrames

    
def NetworkBurstMetric(filename, wellID, path, threshold, percentage_AE=0.5, time_window_seconds=0.25, n_min_spikes=5, ISI_max_seconds=0.1):
    """Extract network burst metrics from BRW file.
    
    Args:
        filename (str): path to the BRW file
        wellID (str): identifier of the selected well
        path (str): path to directory containing pre-processed channel data (.npy files)
        threshold (float): spike rate threshold for active electrode detection
        percentage_AE (float, optional): percentage of active electrodes required for network burst. Defaults to 0.5.
        time_window_seconds (float, optional): time window for network burst detection in seconds. Defaults to 0.25.
        n_min_spikes (int, optional): minimum number of spikes to define a burst. Defaults to 5.
        ISI_max_seconds (float, optional): maximum interspike interval within burst in seconds. Defaults to 0.1.

    Returns:
        tuple: Contains the following elements:
            - network_burst (array): network burst indicator per time window
            - n_NB (int): total network burst count
            - NB_rate (float): network burst rate
            - NB_duration (array): network burst duration per burst in seconds
            - NB_spikes (array): spike count per network burst
            - NB_spikes_percentage (array): percentage of total spikes per network burst
            - NB_ISI (array): inter-spike interval within network bursts in ms
            - NB_INBI (array): inter-network-burst intervals in ms
            - NB_INBI_var (float): inter-network-burst interval variance
    """
    Active_electrodes_ID, bursts, n_bursts, bursts_rate, bursts_n_spikes, bursts_spikes_percentage, bursts_ISI, bursts_ISI_average, bursts_n_spikes_average, bursts_duration, bursts_duration_average, IBI, IBI_average, IBI_std, bursts_well, n_bursts_well, bursts_rate_well, bursts_duration_well_average, bursts_duration_well_std, IBI_well, IBI_well_average, IBI_well_std, bursts_n_spikes_well, bursts_n_spikes_well_average, bursts_n_spikes_well_std, bursts_ISI_well, bursts_ISI_well_average, bursts_ISI_well_std, bursts_spikes_percentage_well, SamplingRate, NumFrames = BurstsMetric(filename, wellID, path, threshold, n_min_spikes, ISI_max_seconds)
    time_window_frames = int(time_window_seconds*SamplingRate)
    
    bursts_spikes = np.zeros((len(Active_electrodes_ID), NumFrames))
    for ch in Active_electrodes_ID:
        idx = np.where(Active_electrodes_ID==ch)[0]
        for i in range(len(bursts[ch])):
            bursts_spikes[idx, bursts[ch][i]] = 1
    spikes_tot = np.sum(bursts_spikes, axis = 0) 
    spikes_tot[spikes_tot>0] = 1
    n_spikes_tot = np.sum(spikes_tot)        
    aux = int(NumFrames/time_window_frames) 
    NB_spikes_matrix = spikes_tot[0:aux*time_window_frames].reshape(aux, time_window_frames)   
    NB_spikes = np.sum(NB_spikes_matrix, axis=1) 
    NB_duration = np.zeros(NB_spikes_matrix.shape[0])
    for i in range(NB_spikes_matrix.shape[0]):
        idx = np.where(NB_spikes_matrix[i]>0)[0]
        if len(idx)>1:
            start = idx[0]
            end = idx[-1]
            NB_duration[i] = (end-start)/SamplingRate
        else:
            NB_duration[i] = 0    
    result = np.zeros((len(Active_electrodes_ID), aux))
    result[:, 0:aux]  = (bursts_spikes[:, 0:aux*time_window_frames].reshape(result.shape[0], aux, time_window_frames)).sum(axis=2)   
    result[result>0] = 1 
    network_burst = np.sum(result, axis=0)
    idx_NB = np.where(network_burst>=len(Active_electrodes_ID)*percentage_AE)[0]
    n_NB = len(idx_NB) 
    NB_rate = n_NB/NumFrames*SamplingRate
    NB_spikes_percentage = NB_spikes/n_spikes_tot*100
    NB_ISI = NB_duration*1000/NB_spikes  
    NB_INBI = np.zeros(len(idx_NB)-1) 
    for i in range(len(NB_INBI)):
        start = np.where(NB_spikes_matrix[idx_NB[i]]>0)[0][-1] 
        end = np.where(NB_spikes_matrix[idx_NB[i+1]]>0)[0][0]+time_window_frames*(idx_NB[i+1]-idx_NB[i])
        NB_INBI[i] = (end-start)/SamplingRate*1000 
    NB_INBI_var = np.std(NB_INBI) 
    print('---') 

    return network_burst, n_NB, NB_rate, NB_duration[idx_NB], NB_spikes[idx_NB], NB_spikes_percentage[idx_NB], NB_ISI[idx_NB], NB_INBI, NB_INBI_var

