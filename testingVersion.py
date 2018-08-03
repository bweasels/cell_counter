import cv2
import os
import gc
import numpy as np

def main():
	#Load the images and create a bgr and greyscale copy
	bgr_bf_image = cv2.imread('bf_test.tif',1)
	gs_bf_image = cv2.cvtColor(bgr_bf_image, cv2.COLOR_BGR2GRAY)
	bgr_fitc_image = cv2.imread('fitc_test.tif', cv2.IMREAD_COLOR)
	gs_fitc_image = cv2.cvtColor(bgr_fitc_image, cv2.COLOR_BGR2GRAY)
	
	####----Analyzeze the bf image-----#####
	#count the overall number of droplets
	circles = brightFieldAnalyze(gs_bf_image)
	
	#draw circles on the fitc Image for qc
	#for i in circles[0,:]:
	#	cv2.circle(bgr_fitc_image, (i[0],i[1]),i[2],(0,255,0),2)
	#	cv2.circle(bgr_fitc_image, (i[0],i[1]),2,(0,0,255),3)
	
	#output number of circles (again QC shit)
	
	
	#display & save the image (moar QC yo)
	#cv2.imshow('image', bgr_fitc_image)
	#cv2.waitKey()
	#cv2.imwrite('coloredArt.tif', bgr_fitc_image)
	
	#####-----Analyze the fitc image-----#####
	pos_count = fitcAnalyze(gs_fitc_image)
	
	print('Ratio of Glowing Cells to droplets is %s/%s' % (pos_count, len(circles[0])))
	return()

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
	circles = np.uint16(np.around(circles))
	
	return(circles)

def fitcAnalyze(img):
	
	#Create a gradient map image by shrinking very small and returning to original size
	mapImg = cv2.resize(img, None, fx = 0.01, fy = 0.01, interpolation = cv2.INTER_AREA)
	mapImg = cv2.resize(mapImg, (4908, 3264))
	
	#correct for a gradient by subtracting
	gc_img = cv2.subtract(img, mapImg)

	#threshold the gc_image for contour finding
	ret, thresh_img = cv2.threshold(gc_img, 55, 255, cv2.THRESH_BINARY)
	
	#find the contours
	image, contours, __ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)	
	
	positiveCount = 0
	for c in range(len(contours)):
		area = cv2.contourArea(contours[c])
		if area < 600 and area > 10:
			cv2.drawContours(gc_img,contours, c, (255,255,0), -1)
			positiveCount += 1
	
	
	return(positiveCount)
	
def settingUpForWaterShed():
	
	#determine confident background
	sure_bg = cv2.dilate(opening, kernel, iterations=3)
	
	#Find sure foreground area
	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
	ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255,0)
	
	
	#Find unsure region (boudnary between sure_bg and sure_fg
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg, sure_fg)

	#Marker labelling for watershedding
	ret, markers = cv2.connectedComponents(sure_fg)
	
	#Add one to all labels so that sure background is not 0, but 1
	markers = markers+1
	
	# Now mark unknown region with 0
	markers[unknown==255] = 0
	
	#reset markers as set of boundaries found by watershed
	markers = cv2.watershed(bgr_image, markers)
	
	#set boundary as black and everything else white on the greyscale image 
	thresh_img[markers==-1] = [0]
	thresh_img[markers==1] = [0]
	
	#Invert the image because contours uses white as background and black as stuff
	thresh_img = cv2.bitwise_not(thresh_img)
	
	#produces the array of contours for the image
	img, contours, __ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)	
	
	#creates a drawn on image for quality control
	thresh_img = cv2.drawContours(thresh_img,contours, -1, [255], 3)
	
	contourAreas = [None]*len(contours)
	for c in range(len(contours)):
		area = cv2.contourArea(contours[c])
		if (area > 10000 and area < 30000):
			contourAreas[c] = area
	colonies = filter(None, contourAreas)
	print(len(colonies))
	#cv2.imwrite('output.tif', image)
	
	return()

		
if __name__ == "__main__":
	main()
