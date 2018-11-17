########################################
#file: cell_signal_counter.py
#input: images with the signal in each channel
#output: counts of the number of droplets and the number of cells within droplets in each channel
#output: also the intensity of the signal
########################################
#!/usr/bin/python

import cv2
import os
import numpy as np
import sys, getopt

def read_img(addr):
	#Load the brightfield image
	image = cv2.imread(addr, 0)
	
	#threshold for improved nn performance
	#_, thresh_img =  cv2.threshold(image, 138, 255, cv2.THRESH_BINARY)
	
	return(image)

def generate_training_set(bf_img):

	###Initial parameter setting
	w = 12 #1/2 the width of the sliding window (ie a w of 12 = 24x24 window)
	y = 0 #sets start of x and y @ 0,0
	x = 0
	step = 8 #num of pixels moved per frame
	root = os.path.abspath('.')
	
	#Gets the size of the image
	(row, column) = bf_img.shape
	
	#Counts the number of rows and columns that the sliding window will generate
	nRows = (row - w)/step
	nColumns = (column - w)/step
	
	#create the positive and negative folders
	positiveFolder = os.path.join(root, 'positiveSet')
	negativeFolder = os.path.join(root, 'negativeSet')
	
	#make the positive and negative folders
	if not os.path.exists(positiveFolder):
		os.makedirs(positiveFolder)
	if not os.path.exists(negativeFolder):
		os.makedirs(negativeFolder)
	
	###voting on the training set!
	
	#go through the stepped columns and rows
	for i in range(nColumns):
		
		#set x to change with the for loop with size step & convert to an int
		x = np.abs((i-1)*step)
		
		for j in range(nRows):
			
			#convert y to numpy friendly ints and increment with the inner for loop
			y = np.abs((j-1)*step)
			outBounds = False
			
			#create the image square and omit out of bounds images
			crop = bf_img[y-w:y+w, x-w:x+w]
			if (y-w < 0) or (y+w > row) or (x-w < 0) or (x+w > column):
				outBounds = True

			#if not off the screen then send to me for judging as positive or negative
			if not outBounds:
				#perCompleted = int(((((i-1)*nRows)+j)/nColumns*nRows)*100)
				#print perCompleted
				#resize as larger so that I can actually see it
				tempCrop = cv2.resize(crop, (300, 300))
				#imageName = "crop {}% finished".format(perCompleted)
				cv2.imshow('crop', tempCrop)
				input = cv2.waitKey()

				#take input y or n and move the image to the positive or negative folder
				if (input == ord('y')):
					imFile = os.path.join(positiveFolder, (str(y) + '_' + str(x) + '.jpg'))
					cv2.imwrite(imFile, crop)
				elif (input == ord('n')):
					imFile = os.path.join(negativeFolder, (str(y) + '_' + str(x) + '.jpg'))
					cv2.imwrite(imFile, crop)
				else:
					print('incorrect entry, image thrown out\n')
	return()
	
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

def main(argv):
	
	bf_addr = 'images/droplet_bf.tif'
	bf_img = read_img(bf_addr)
	
	try:
		opts, args = getopt.getopt(argv,'t')
	except getopt.GetoptError:
		print 'cell_signal_counter.py -t to generate training set \ncell_signal_counter.py to run normal stuff'
		sys.exit()
		
	for opt, arg in opts:
		if opt == '-t':
			generate_training_set(bf_img)
			sys.exit()
	
	cv2.imshow('thresholded bf image', bf_img)
	cv2.waitKey()
	
	return()

if __name__ == "__main__":
	main(sys.argv[1:])
