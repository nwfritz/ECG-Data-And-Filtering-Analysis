# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:05:52 2023

File Name: project3_module.py

Project 3 module holds functions that are used in project 3 script. A function is used to load and plot initial ecg data. Create a filter based on the
artifacts found in plotted ECG data to create frequency and impulse response. Data is filtered for analysis, pulling heartbeats from the data to find
the heartbeat variability to calculate and plot heart rate mean power variability. The plots and values created are used for analysis to answer the question 
if frequency content can be used to assess autonomic nervous system activity.

@author: Tori Grosskopf and Nathan Fritz
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, freqz, find_peaks
from scipy.fft import rfft, irfft, rfftfreq, fftshift
#%% Part 1: Load and Plot Raw Data

def load_graph_data(file_name, subplot_value, title):
    """
    Function load_graph_data loads ecg data, crops it, creates an associated time array
    and plots a region of the data in the time domain 

    Parameters
    ----------
    file_name : string
        Represents the name of the file containing a type of data
    subplot_value : int 
        Represnt the sub plot number that will be created
    title : string
        Represents the title associated with specific data

    Returns
    -------
    data : array of floats (x, 1), where x is cropped amount of data that represents 300 seconds (5 minutes)
        represents the ecg signal data for an event

    """
    # define fs
    fs = 500 # given from Dr. J's Code
    # load_data
    data = np.loadtxt(file_name)
    # crop data to be 5 minutes, 300 seconds long 
    start_time = 5
    end_time = 305
    data = data[start_time*fs:end_time*fs]
    # create time array based on length of the data  
    time = np.arange(0, len(data)/fs, 1/fs)
    # create sub_plot based on value of subplot value
    plt.subplot(5,1,subplot_value)
    # create plot
    # changing the data to mV, converting from bits to mV
    data = (data * 5) / 1023
    plt.plot(time, data)
    # annotate the plot 
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage(mV)')
    plt.title(title)
    # crop to see only 5 seconds
    plt.xlim(255,260)
    plt.grid()
    # return data 
    return data 

def concatenated_graph_data(data_tuple, subplot_value):
    """
    Function concatenated_graph_data concatenates all ecg data collected into a singal
    array that is 4 times the size of an individual ecg collection event. The fucntion 
    creates an associated time array and plots the concatenated ecg data of all events.

    Parameters
    ----------
    data_tuple : tupple of arrays size w, where w is the number of data collection events
        represents a tuple that stores a different array for each ecg data collection event
    subplot_value : int 
        represnt the sub plot number that will be created

    Returns
    -------
    concatenated_data : array of floats size (t, 1), where t is w * (300 seconds worth of ecg data at provided fs)
        represents the concatenated data of all ecg data collection events

    """
    # concatenate all the provided data
    concatenated_data = np.concatenate(data_tuple)
    # define fs
    fs = 500 # given from Dr. J's Code
    # load_data
    # create time array based on the length of the concatenated data and fs 
    time = np.arange(0, len(concatenated_data)/fs, 1/fs)
    # create sub_plot
    plt.subplot(5,1,subplot_value)
    # create plot
    plt.plot(time, concatenated_data)
    # annotate the plot 
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage(mV)')
    plt.title('Concatenated ECG Data')
    plt.grid()
    # return concatenated data 
    return concatenated_data

#%% Part 2: Filter Data

# design bandpass filter for ecg data
def creat_filter_data(data, low_cutoff, high_cutoff, fs, order):
    """
    Function my_filter creates a butterworth filter to filter collected ecg data

    Parameters
    ----------
    data : array of floats (x, 1), where x is cropped amount of data that represents 300 seconds (5 minutes)
        represents the ecg signal data for an event
    low_cutoff : int
        represents the low cut off value for a butterworth filter
    high_cutoff : int
        represents the high cut off value for a butterworth filter
    fs : int
        represents the sampling frequency in which the data was collected
    order : int
        represents the order in which the butterworth filter will be created

    Returns
    -------
    numerator_polynomials : array of floats size (q,1) where q is determined by the filter type, cut off frequencies, and filter order
        Represents the coefficients of the polynomial in the numerator of the transfer function
    denominator_polynomials : array of floats size (q,1) where q is determined by the filter type, cut off frequencies, and filter order
        Represents the coefficients of the polynomial in the denominator of the transfer function
    filtered_data : array of floats size (e, 1) where e is equal to x(amount of data to represent 300 seconds)
        Represents ecg data for a collection event, now filtered by the butterworth filter

    """ 
    # find nyquist value as half of fs
    nyquist = 0.5 * fs
    # define range of frequency cut offs as cut off devided by nyquist to get perfect sampling and keep all necessary data
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    # creat butterworth filter, get numerator and denominator polynomials in return 
    numerator_polynomials, denominator_polynomials  = butter(order, [low, high], btype = 'bandpass')
    # filter data
    filtered_data = filtfilt(numerator_polynomials, denominator_polynomials, data)
    # return numerator and denominator polynomials, and filtered data
    return numerator_polynomials, denominator_polynomials, filtered_data

# plot frequency response
def plot_frequency_response(numerator_polynomials, denominator_polynomials, fs, subplot_value):
    """
    Function plot_frequency_response calculates the frequency response in the frequency domain and plots it in a subplot

    Parameters
    ----------
    numerator_polynomials : array of floats size (q,1) where q is determined by the filter type, cut off frequencies, and filter order
        Represents the coefficients of the polynomial in the numerator of the transfer function
    denominator_polynomials : array of floats size (q,1) where q is determined by the filter type, cut off frequencies, and filter order
        Represents the coefficients of the polynomial in the denominator of the transfer function
    fs : int
        Represents the sampling frequency of collected data. Constant: 500 determined from Doctor Jangraws code to collect data.
    subplot_value : int 
        represnt the sub plot number that will be created

    Returns
    -------
    computed_frequency : array of floats size (s, 1), where s is determined by the length of numerator and denominatror polynomials
        Represents the the frequencies in which each point of frequency response was calculated
    frequency_response : array of floats size (u, 1), where u is the same length of s
        Represents the frequency response of the created filter

    """
    # use scipy.signal.freqz() to compute the frequency response of a digital filter,
    # returns computed freuqency which is the frequencies in which the frequency response was calculated, and the frequency response
    computed_frequency, frequency_response = freqz(numerator_polynomials, denominator_polynomials, whole=False, plot=None, fs=fs)
    # create new subpplot based on the value of subplot_value
    plt.subplot(1,2,subplot_value)
    plt.plot(computed_frequency,frequency_response)
    # annotate plot
    plt.title('Frequency Response')
    plt.xlabel('frequency (Hz)')
    # need a better??
    plt.ylabel('Magnitude (mV)')
    plt.grid()
    return computed_frequency, frequency_response

    
# plot impulse response of filter
def plot_impulse_response(computed_frequency, frequency_response, subplot_value):
    """
    Function plot_impulse_response calculates and plots the impulse response of the filter based on the calculated frequency_response
    
    Parameters
    ----------
    computed_frequency : array of floats size (s, 1), where s is determined by the length of numerator and denominatror polynomials
        Represents the the frequencies in which each point of frequency response was calculated
    frequency_response : array of floats size (u, 1), where u is the same length of s
        Represents the frequency response of the created filter
    subplot_value : int 
        Represents the sub plot number that will be created

    Returns
    -------
    None.

    """
    # define fs as the sampling rate in which the data was collected
    fs = 500 
    impulse_response = irfft(frequency_response, fs)
    impulse_response = fftshift(impulse_response)
    # define a window and half window to crop it
    window = 1
    half_window = window / 2
    # croppng the impulse response to be in the 1 second
    impulse_response_cropped = impulse_response[int(len(impulse_response)//2 - half_window * fs):int(len(impulse_response)//2+half_window*fs)]
    # create time array based on the impulse response, creating a duration of 1s
    time_impulse_response = np.arange(-0.5, len(impulse_response_cropped) / fs - 0.5, 1 / fs)
    
    plt.subplot(1,2,subplot_value)
    plt.plot(time_impulse_response, impulse_response_cropped)
    # annotate plot
    plt.title('Impulse Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (A.U.)')
    plt.grid()
    
def plot_comparison(time, original_data, filtered_data, subplot_value, title):
    """
    Function plot_comparison plots original and filtered data for a collection activity in the same subplot.
    
    Parameters
    ----------
    time : array of floats size (j, 1), where j is the length of time for a data collection times the sampling frequency the data was collected at
        Represents the time of the ecg data
    original_data : array of floats size(g, 1) where g is the same size as j
        Represents the voltages of the ecg data
    filtered_data : array of floats size(d, 1) where d is the same size as j
        Represents the voltages of the filtered ecg data
    subplot_value : int 
        Represents the sub plot number that will be created
    title : string
        Represents the title associated with specific data

    Returns
    -------
    None.

    """
    # create subplot based on subplot value
    plt.subplot(2,2,subplot_value)
    # plot original and filtered data
    plt.plot(time, original_data, label = 'Original Signal', color = 'magenta')
    plt.plot(time, filtered_data, label = 'Filtered Signal', color = 'mediumseagreen')
    # annotate the plot
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)') # probaly need to change
    plt.title(title)
    plt.grid()
    plt.legend()
    # limit the plot to see comparitive
    plt.xlim(255, 260)


def plot_heart_beats(filtered_data, subplot_value, fs, title):
    """
    Function plot_heart_beats determines the indexes in which a heart beat takes place, determines the voltage and time of those indeces and plots
    them on top of filtered ecg data to show when heart beats occur.

    Parameters
    ----------
  filtered_data : array of floats size(d, 1) where d is the same size as j
       Represents the voltages of the filtered ecg data.
    subplot_value : int 
        Represents the sub plot number that will be created
    fs : int
        Represents the sampling frequency of collected data. Constant: 500 determined from Doctor Jangraws code to collect data.
   title : string
       Represents the title associated with specific data

    Returns
    -------
    peaks : array of ints size (k, 1), where k is the number of heart beats in the filtered ecg data
        Represents the index values of the filtered data which is a heart beat
    """
    # threshold determined from looking at filtered data in figure 3 and trial and error
    # define threshold, height, and distance
    threshold = 0
    height = 0.22
    distance = 170
    # create time array based on filtered data
    time = np.arange(0, len(filtered_data)/fs, 1/fs)
    # determine the peaks that represent the heart beats, 
    # based on threshold, height, and distance values
    # peaks is the indexes each heart peak takes place
    peaks = find_peaks(filtered_data, height = height, threshold = threshold, distance = distance)
    # peaks returns a tuple, the actual peaks are 0th index
    peaks = peaks[0]
    # finding the times in which each peak take place by casitng time array by peak indexes
    peaks_time = time[peaks]
    # finding the voltage value of each heart beat by casting filtered data by peak indexes
    heart_beats = filtered_data[peaks]
    # plot filtered data with heart beats on top with a marl
    plt.subplot(2,2,subplot_value)
    plt.plot(time, filtered_data, label = 'Filtered Signal')
    plt.plot(peaks_time, heart_beats, "x", markersize=10, label='Detected Heart Beats')
    # annotate the plot
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.title(title)
    plt.xlim(255, 260)
    plt.legend()
    plt.tight_layout()
    return peaks
    

def determine_hrv(filtered_data, fs, peaks):
    """
    Function determine_hrv uses found heart beats of a filtered data set and finds the variability between each set of heart beats.

    Parameters
    ----------
    filtered_data : array of floats size(d, 1) where d is the same size as j
         Represents the voltages of the filtered ecg data.
    fs : int
        Represents the sampling frequency of collected data. Constant: 500 determined from Doctor Jangraws code to collect data.
    peaks : array of ints size (k, 1), where k is the number of heart beats in the filtered ecg data
        Represents the index values of the filtered data which is a heart beat

    Returns
    -------
    peaks_time : array of floats size (c,1), where c is the same size as k
        Represents the time values for each peak/heart beat
    beat_intervals : array of floats size (o, 1), where o is k - 1
        Represents the time between heart beats for a data set
    heart_rate_variability : float
        Represents the standard deviation of the heart beat intervals, showing heart beat variability for a data set

    """
    # create a time array based on filtered_data
    time = np.arange(0, len(filtered_data)/fs, 1/fs)
    # find the time (s) in which each peak takes place
    peaks_time = time[peaks]
    # determined heart rate variability
    # creat an array of zeros to store beat intervals
    beat_intervals = np.zeros(len(peaks_time))
    # loop through each peak time too find the difference of heart beats
    for heart_beat_index in range(len(peaks_time) -1):
        # find the difference between two beats 
        difference = peaks_time[heart_beat_index+1] - peaks_time[heart_beat_index]
        # add the difference to the beat_interval at the specific interval index
        beat_intervals[heart_beat_index] = difference
    # deleting the last time index which is zero
    beat_intervals = beat_intervals[:-1]
    # finding the standard deviation of the beat_intervals
    heart_rate_variability = np.std(beat_intervals)
    # return peaks, their associated time, heart rate variability, and intervals of each beat
    return peaks_time, beat_intervals, heart_rate_variability

def determine_interpolated_timecourse(beat_intervals, time, peaks):
    """
    Function determine_interpolated_timecourse creates an interplated timecourse for an ecg collection event 

    Parameters
    ----------
    beat_intervals : array of floats size (o, 1), where o is k - 1
        Represents the time between heart beats for a data set
    time : array of floats size (j, 1), where j is the length of time for a data collection times the sampling frequency the data was collected at
        Represents the time of the ecg data
    peaks : array of ints size (k, 1), where k is the number of heart beats in the filtered ecg data
        Represents the index values of the filtered data which is a heart beat

    Returns
    -------
    interpolated_timecourse : array of floats size (p, 1), where p is the time taken of an ecg event times the sampling frequency
        Represents the interpolated time of the ecg voltage of a data collection type

    """
    # constant given as the time between each sample
    dt = 0.10
    # create a time array for interpalation 
    interp_time = np.arange(0, len(time) * dt, dt)
    # crop normal time array to get the time are each peak, ignoring first one so they are same lenght
    time = time[peaks][1::]
    # create interpolated timecourse using linear interpolation 
    interpolated_timecourse = np.interp(interp_time, time, beat_intervals)
    # return interpolated timecourse
    return interpolated_timecourse

def calculate_and_plot_mean_powers(interpolated_timecourse, dt, subplot_value, title):
    """
    Function calculate_and_plot_mean_powers calculates the power and frequency of a data collection event,
    and plots the high and low frequency regions on the same plot. Calculates and plots ratio of high and low frequencies.

    Parameters
    ----------
    interpolated_timecourse : array of floats size (p, 1), where p is the time taken of an ecg event times the sampling frequency
        Represents the interpolated time of the ecg voltage of a data collection type
    dt : float
        Represents the intervals in which data is collected
    subplot_value : int 
        Represents the sub plot number that will be created
    title : string
        Represents the title associated with specific data

    Returns
    -------
    frequency_ratio : float
        Represents the ratio of mean high frequency power over low frequency power

    """
    # define high and low frequency regions
    
    low_frequency = [0.04, 0.15]
    high_frequency = [0.15, 0.4]
    # find the fft of interpolated signal
    fft = rfft(interpolated_timecourse)
    # find magnitude of fft
    fft_magnitude = np.abs(fft)
    # calculate power of fft
    power = np.square(fft_magnitude)
    # determine frequency of interpolated signal
    freq = rfftfreq(len(interpolated_timecourse), dt)
    # casting the mean power to get high and low frequency power arrays
    low_freq_indices = np.where((freq >= low_frequency[0]) & (freq < low_frequency[1]))[0]
    high_freq_indices = np.where((freq >= high_frequency[0]) & (freq < high_frequency[1]))[0]
    # find the ratio between high and low frequency areas
    frequency_ratio = np.mean(power[low_freq_indices]) / np.mean(power[high_freq_indices])
    
    # create new subplot based on subplot value
    plt.subplot(2,2,subplot_value)
    # plot figure
    plt.plot(freq[low_freq_indices], power[low_freq_indices], color = 'green', label = 'Low freq mean power')
    plt.plot(freq[high_freq_indices], power[high_freq_indices], color = 'pink', label = 'High freq mean power')
    # fill between the line and the x-axis
    plt.fill_between(freq[low_freq_indices], power[low_freq_indices], color='green', alpha=0.3)
    plt.fill_between(freq[high_freq_indices], power[high_freq_indices], color='pink', alpha=0.3)
    # annotate plot
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (mV^2)')
    plt.title(title)
    plt.grid()
    plt.legend()
    # return frequency ratio
    return frequency_ratio

