########################################
#file: cell_signal_counter.py
#input: images with the signal in each channel
#output: counts of the number of droplets and the number of cells within droplets in each channel
#output: also the intensity of the signal
########################################

#Program Structure:
#cell_signal_counter.py - calls all the files
##Read_img.py - reads image, thresholds
##|
##V
##Generate_Sliding_window.py - generates a sliding window <-----------------------|
##|                                                                               |
##Hands sliding window to                                         repeat over all frames in the window
##|                                                                               |
##V                                                                               |
##Droplet_detect.m - NN that will determine if there's a droplet in the window ---|
##|
##cell_signal_counter.py generates a mask based off of the aforementioned loop
##
