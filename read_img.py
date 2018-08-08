#####################################
###FileName: Read_img.py
###Input: Large bright_field image
###Output: thresholded and scaled image
#####################################
import cv2
import os
import numpy as np

def main():
	#Load the images and create a sliding window for Positive negative classification
	image = cv2.imread('images/droplet_bf.tif', 0)
	(row, column) = image.shape
	_, thresh_img =  cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
	return()
	
	
if __name__ == "__main__":
	main()