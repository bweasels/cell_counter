import cv2
import os
import numpy as np
import csv

def main():
	#Load the images and create a sliding window for Positive negative classification
	
	#Steps for the sliding window
	w = 15
	y = 0
	x = 0
	step = 10
	root = os.path.abspath('.')
	
	image = cv2.imread('images/droplet_bf.tif', 0)
	(row, column) = image.shape
	trainingDim = (30, 30)
	
	#create the positive and negative folders
	positiveFolder = os.path.join(root, 'positiveSet')
	negativeFolder = os.path.join(root, 'negativeSet')
	
	#make the positive and negative folders
	if not os.path.exists(positiveFolder):
		os.makedirs(positiveFolder)
	if not os.path.exists(negativeFolder):
		os.makedirs(negativeFolder)
	
	nRows = (row - w)/step
	nColumns = (column - w)/step

	#go through the stepped columns and rows
	for i in range(nColumns):
		
		#set x to change with the for loop with size step & convert to an int
		x = np.abs((i-1)*step)
		
		for j in range(nRows):
			
			#convert y to numpy friendly ints and increment with the inner for loop
			y = np.abs((j-1)*step)
			outBounds = False
			
			#create the image square and omit out of bounds images
			crop = image[y-w:y+w, x-w:x+w]
			if (y-w < 0) or (y+w > row) or (x-w < 0) or (x+w > column):
				outBounds = True

			#if not off the screen then send to me for judging as positive or negative
			if not outBounds:
			
				#resize as larger so that I can actually see it
				tempCrop = cv2.resize(crop, (300, 300))
				cv2.imshow('crop', tempCrop)
				input = cv2.waitKey()

				#take input y or n and move the image to the positive or negative folder
				if (input == ord('y')):
					imFile = os.path.join(positiveFolder, (str(y) + str(x) + '.jpg'))
					cv2.imwrite(imFile, crop)
				elif (input == ord('n')):
					imFile = os.path.join(negativeFolder, (str(y) + str(x) + '.jpg'))
					cv2.imwrite(imFile, crop)
				else:
					print('incorrect entry, image thrown out\n')
	return()
	
	
if __name__ == "__main__":
	main()