import cv2
import os
import numpy as np
import csv

def brightFieldAnalyze(img):
	#####----Analyze the bf image----#####
	#Threshold the greyscale img and blur
	ret,thresh_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
	thresh_img = cv2.medianBlur(thresh_img, 5)
	
	#eroding via Morphology Ex to smoothen out the image
	kernel = np.ones((3,3), np.uint8)
	opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations =2)
	
	#addn blurring to remove any remaining artifacts
	opening = cv2.medianBlur(opening, 5)
	
	#calculate hough circles
	circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, dp=2, minDist=100, param1=70, param2=50, minRadius=50, maxRadius=100)
	
	#convert circles to unsigned 16bit ints so can be used to draw new circles
	if circles is not None:
		circles = np.uint16(np.around(circles))
	
	return(circles)

def main():
	#Load the images and create a sliding window for Positive negative classification
	
	#Steps for the sliding window
	w = 10
	y = 0
	x = 0
	step = 6
	root = os.path.abspath('.')
	#Y = vote yes, N = vote no
	
	#open image and extract row and column counts
	image = cv2.imread('images/droplet_bf.tif', 0)
	#image = cv2.resize(image, (0,0), fx = 0.2, fy = 0.2)
	(row, column) = image.shape
	trainingDim = (20, 20)
	
	#create the positive and negative folders
	positiveFolder = os.path.join(root, 'positiveSet')
	negativeFolder = os.path.join(root, 'negativeSet')
	
	#make the positive and negative folders
	if not os.path.exists(positiveFolder):
		os.makedirs(positiveFolder)
	if not os.path.exists(negativeFolder):
		os.makedirs(negativeFolder)
	
	#run hough circles on the whole image
	#circles = brightFieldAnalyze(image)
	
	#convert to numpy friendly intss
	#if circles is not None:
	#	circles = np.round(circles[0, :]).astype('int')
		
		#go through the circels and get the x, y and radius values
	#	for (x, y, r) in circles:
	#		outBounds = False
			#Throw out out of bounds images
	#		if (y-w < 0) or (y+w > row) or (x-w < 0) or (x+w > column):
	#			outBounds = True
			#crop the image and display the inbounds ones
	#		crop = image[y-w:y+w, x-w:x+w]
	#		if not outBounds:
	#			imageFile = os.path.join(positiveFolder, (str(x) + str(y) + '.jpg'))
	#			cv2.imwrite(imageFile, crop)
				
	#set the number of rows and columns so that the negative image won't fall off
	
	nRows = (row - w)/step
	nColumns = (column - w)/step

	y = 0
	x = 0

	#go through the stepped columns and rows
	for i in range(nColumns):
		
		#convert x to numpy friendly ints
		x = np.abs((i-1)*step)
		for j in range(nRows):
			
			#convert y to numpy friendly ints
			y = np.abs((j-1)*step)
			
			outBounds = False
			
			#create the image square and omit out of bounds images
			crop = image[y-w:y+w, x-w:x+w]
			if (y-w < 0) or (y+w > row) or (x-w < 0) or (x+w > column):
				outBounds = True

			#if not off the screen then send to me for judging as positive or negative
			if not outBounds:
			
				#resize as larger so that I can actually see it
				tempCrop = cv2.resize(crop, (200, 200))
				cv2.imshow('crop', tempCrop)
				input = cv2.waitKey()

				if (input == ord('y')):
					imFile = os.path.join(positiveFolder, (str(y) + str(x) + '.jpg'))
					cv2.imwrite(imFile, crop)
				elif (input == ord('n')):
					imFile = os.path.join(negativeFolder, (str(y) + str(x) + '.jpg'))
					cv2.imwrite(imFile, crop)
				else:
					print('incorrect entry, image thrown out\n')
#				cv2.imwrite(imageFile, crop)
	return()
	
	
if __name__ == "__main__":
	main()