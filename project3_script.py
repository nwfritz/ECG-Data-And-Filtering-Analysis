# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:06:21 2023

File Name: project3_script.py

In this project collected ECG data at four distinct data collection events is used to investigate whether the frequency content of found heart rate variability
can be used to accurately assess autonomic nervous system activity. To make this happen, data was collected, imported, and filtered to remove the noise that came from data collection.
Peaks/heartbeats were found based on the filtered data and used to determine heart rate variability. With heart rate variability, mean power in the frequency domain was used to 
show heart rate variability frequency band power. Figures were created along the way to be used in analysis to answer the question if frequency content can be used to assess 
autonomic nervous system activity.

@authors: Nathan Fritz and Tori Grosskopf
"""
import numpy as np
from matplotlib import pyplot as plt
import project3_module as p3m
from scipy.signal import butter, filtfilt, impulse, find_peaks


# %% Part 1: Load and Plot Raw Data
# create overarching figure for 5 plots
plt.figure(1, clear = True)
fs = 500
# Loading in and plotting each data set using function from Project3 Module
sitting_at_rest_data = p3m.load_graph_data('sitting_at_rest.txt', 1, title = 'Sitting at Rest ECG Data')
relaxing_activity_data = p3m.load_graph_data('relaxing_activity.txt',2, title = 'Relaxing Activity ECG Data')
mentally_stressful_data = p3m.load_graph_data('mentally_stressful.txt',3, title = 'Mental Stress Activity ECG Data')
physically_stressful_data = p3m.load_graph_data('physically_stressful.txt',4, title = 'Physical Stress Activity ECG Data')
# create a tuple to hold all collection event datas
data_tuple = ((sitting_at_rest_data, relaxing_activity_data, mentally_stressful_data, physically_stressful_data))
# concatenate all the data taken and plotting it
concatenated_data = p3m.concatenated_graph_data(data_tuple, 5)
# annotate plot
#plt.tight_layout()
plt.savefig('raw_event_data.png')


#%% Part 2: Filter Data

# define cutoff frequencies in Hz
low_cutoff = 5
high_cutoff = 50

# define sampling rate
fs = 500 # given from Dr. Js code

# filter data, get numerator_polynomials and denominator_polynomials of filter
numerator_polynomials, denominator_polynomials, filtered_resting_data = p3m.creat_filter_data(sitting_at_rest_data, low_cutoff, high_cutoff, fs, order = 3)
numerator_polynomials, denominator_polynomials, filtered_relaxing_data = p3m.creat_filter_data(relaxing_activity_data, low_cutoff, high_cutoff, fs, order = 3)
numerator_polynomials, denominator_polynomials, filtered_mental_stress_data = p3m.creat_filter_data(mentally_stressful_data, low_cutoff, high_cutoff, fs, order = 3)
numerator_polynomials, denominator_polynomials, filtered_physical_stress_data = p3m.creat_filter_data(physically_stressful_data, low_cutoff, high_cutoff, fs, order = 3)

# define time and blank impulse signal
time = np.arange(0,len(filtered_relaxing_data)/fs, 1/fs)
impulse_signal = np.zeros_like(time)
# create figure 2 to plot in
plt.figure(2, clear = True)
# plot impulse and frequency response 
computed_frequency, frequency_response = p3m.plot_frequency_response(numerator_polynomials, denominator_polynomials, fs, 2)
p3m.plot_impulse_response(computed_frequency, frequency_response, 1)
plt.savefig('impulse_and_frequency_response.png')

# creat figure 3 to plot comparitive graph
# comparitive graph of original signals and filtered signals
plt.figure(3, clear = True)
 # plot comparitive subplots
p3m.plot_comparison(time, sitting_at_rest_data, filtered_resting_data, 1, title = 'Resting: Original vs Filtered ECG Data')
p3m.plot_comparison(time, relaxing_activity_data, filtered_relaxing_data, 2, title = 'Relaxing: Original vs Filtered ECG Data')
p3m.plot_comparison(time, mentally_stressful_data, filtered_mental_stress_data, 3, title = 'Mental Stress: Original vs Filtered ECG Data')
p3m.plot_comparison(time, physically_stressful_data, filtered_physical_stress_data, 4, title = 'Physical Stress: Original vs Filtered ECG Data')
plt.savefig('raw_vs_filtered_ecg_data.png')
# %% Part 3
# plot figure 4
plt.figure(4, clear = True)
resting_peaks = p3m.plot_heart_beats(filtered_resting_data, 1, fs, title = 'Resting Heart Beats')
relaxing_peaks = p3m.plot_heart_beats(filtered_relaxing_data, 2, fs, 'Relaxing Heart Beats')
mental_stress_peaks = p3m.plot_heart_beats(filtered_mental_stress_data, 3, fs, 'Mental Stress Heart Beats')
physical_stress_peaks = p3m.plot_heart_beats(filtered_physical_stress_data, 4, fs, 'Physical Stress Heart Beats')
plt.savefig('heart_beats.png')

#%% Part 4: Heart Rate Variability
 
# determine beat times
resting_peaks_time, resting_beat_intervals, resting_heart_rate_variability = p3m.determine_hrv(filtered_resting_data,fs,resting_peaks)
relaxing_peaks_time, relaxing_beat_intervals, relaxing_heart_rate_variability = p3m.determine_hrv(filtered_relaxing_data,fs,relaxing_peaks)
mental_stress_peaks_time, mental_stress_beat_intervals, mental_stress_heart_rate_variability = p3m.determine_hrv(filtered_mental_stress_data, fs,mental_stress_peaks)
physical_stress_peaks_time, physical_stress_beat_intervals, physical_stress_heart_rate_variability = p3m.determine_hrv(filtered_physical_stress_data, fs,physical_stress_peaks)
 
# make bar graph of hrv
all_heart_rate_variability = np.array((resting_heart_rate_variability, relaxing_heart_rate_variability, mental_stress_heart_rate_variability, physical_stress_heart_rate_variability))
 
heart_rate_variability_labels = np.array(('Resting HRV', 'Relaxing HRV', 'Mental Stress HRV', 'Physical Stress HRV'))
bar_colors = np.array(('skyblue', 'salmon', 'lightgreen', 'magenta'))
plt.figure(5, clear=True)
plt.bar(heart_rate_variability_labels, all_heart_rate_variability, color = bar_colors)
# annotate plot
plt.title('Heart Rate Variability(HRV) Comparison')
plt.ylabel('Variability (s)')
# save figure
plt.savefig('heart_rate_variabilities.png')
# calculate interpolated time array for all data
resting_timecourse_interpolation = p3m.determine_interpolated_timecourse(resting_beat_intervals, time, resting_peaks)
relaxing_timecourse_interpolation = p3m.determine_interpolated_timecourse(relaxing_beat_intervals, time, relaxing_peaks)
mentally_stress_timecourse_interpolation = p3m.determine_interpolated_timecourse(mental_stress_beat_intervals, time, mental_stress_peaks)
physically_stress_timecourse_interpolation = p3m.determine_interpolated_timecourse(physical_stress_beat_intervals, time, physical_stress_peaks)

# %% Part 5: Get HRV Frequency Band Power
# define dt
dt = 0.1
# create figure to plot mean power frequency
plt.figure(6, clear = True)
# call function to calculate and plot mean power and frequency bands
resting_frequency_ratio = p3m.calculate_and_plot_mean_powers(resting_timecourse_interpolation, dt, 1, title ='Resting FFT Spectrum')
relaxing_frequency_ratio = p3m.calculate_and_plot_mean_powers(relaxing_timecourse_interpolation, dt, 2, title ='Relaxing FFT Spectrum')
mental_stress_frequency_ratio = p3m.calculate_and_plot_mean_powers(mentally_stress_timecourse_interpolation, dt, 3, title ='Mental Stress FFT Spectrum')
physical_stress_frequency_ratio = p3m.calculate_and_plot_mean_powers(physically_stress_timecourse_interpolation, dt, 4, title ='Physical Stress FFT Spectrum')
# save figure 6
plt.savefig('heart_rate_ratios.png')
# creat arrays of all the collection event frequency ratios
all_heart_rate_frequency_ratio = np.array((resting_frequency_ratio, relaxing_frequency_ratio, mental_stress_frequency_ratio, physical_stress_frequency_ratio))
all_heart_rate_frequency_ratio_labels = np.array(('Resting Ratio', 'Relaxing Ratio', 'Mental Stress Ratio', 'Physical Stress Ratio'))
# create figure to plot heart rate frequency ratios
plt.figure(7, clear =True)
plt.bar(all_heart_rate_frequency_ratio_labels, all_heart_rate_frequency_ratio, color = bar_colors)
# annotate plot
plt.title('Ratios of Mean Power: Low-Freq/High-Freq')
plt.ylabel('Ratio Value (A.U.)')
# save figure 
plt.savefig('frequency_ratios.png')

