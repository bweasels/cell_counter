import cv2
import os
import gc
import sys
import numpy

CELL_MAX_SIZE = 20
DROP_RADIUS = 32

DIAGNOSIS_MODE = False


#####SUPPORT FUNCTIONS FOR FIND DROPLET#####
def cropImg(img, y, x, height, width):
	#Add or subtract the droplet radius 
	Top = y-DROP_RADIUS
	Bottom = y+DROP_RADIUS
	Left = x-DROP_RADIUS
	Right = x+DROP_RADIUS
	
	#If the image goes out of bounds, correct
	if Top < 0:
		Top = 0
	if Left < 0:
		Left = 0
	if Bottom > height:
		Bottom = height
	if Right > width:
		Right = width

	#Crop the image 
	crop_img = img[Top:Bottom, Left:Right]
	
	return(crop_img)

def grossDropletVal(img):
	#Greyscale the image to make things easier
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	height, width = img.shape
	grossPixelVal = 0
	nPixels = 0
	
	##Go through and add up pixel values. If the pixel value is 0 then it has been masked out and can be ignored
	for h in range(height):
		for w in range(width):
			if img[h][w] != 0:
				grossPixelVal += img[h][w]
				nPixels+=1

	return(grossPixelVal) 

#####FINDS DROPLETS AND QUANTS THE PIXEL VALUES IN EACH DROPLET#####
def findDroplets(img, cy5_img):
	height, width = img.shape
	
	#threshold for houghCircles
	ret,thresh_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

	if DIAGNOSIS_MODE:
		cv2.imshow('thresholded image for circle detection', thresh_img)
		cv2.waitKey()
	
	#houghCircles params
	circles = cv2.HoughCircles(thresh_img, cv2.HOUGH_GRADIENT, dp=2, minDist=40, param1=70, param2=50, minRadius=25, maxRadius=40)
	
	####Dianosing the circle detection####
	if DIAGNOSIS_MODE:
		temp_img = img
		temp_circles = circles
		if temp_circles is not None:
			temp_circles = numpy.round(temp_circles[0, :]).astype("int")
			for (x, y, r) in temp_circles:
				cv2.circle(temp_img, (x,y),r,(0,255,255),3)

		cv2.imshow('Detected Circles', temp_img)
		cv2.waitKey()
	####End diagnosis####
	
	#initialize the mask image
	mask_img = numpy.zeros((height, width, 3), numpy.uint8)
	
	##If there are no droplets then return none
	if circles is None:
		nDroplets = 0
		blanked = mask_img
		dropBright = [None]
		return(dropBright, nDroplets, blanked)

	#convert circles to unsigned 16bit ints so can be used to draw new circles
	circles = numpy.uint16(numpy.around(circles))
	nDroplets = len(circles[0])
	dropBright = [None]*nDroplets
	
	#draw circles on the mask and get the cy5 pixel value for each droplet
	####Droplets are 66px x 66px ####
	i = 0
	for circle in circles[0,:]:
		cv2.circle(mask_img, (circle[0],circle[1]),circle[2],(255,255,255),-1)
		crop_img = cropImg(cy5_img, circle[1], circle[0], height, width)
		dropBright[i] = grossDropletVal(crop_img)
		i = i+1

	#mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
	blanked = cv2.bitwise_and(cy5_img, mask_img, mask_img)
	return(dropBright, nDroplets, blanked)

#####WRAPPER FUNCTION THAT ANALYZES THE GIVEN FILES#####
def analyzeFile(filename, root):	
	#get the image path
	BFaddr = filename+'_BF.jpg'
	CY5addr = filename+'_Cy5.jpg'
	print('Opening files: ' + BFaddr + ', ' + CY5addr)
	
	#Load images
	imgAddr = os.path.join(root, BFaddr)
	cy5imgAddr = os.path.join(root, CY5addr)
	img = cv2.imread(imgAddr, 0)
	cy5_img = cv2.imread(cy5imgAddr, 1)
	
	####STEP: FIND DROPLETS BY MASKING EVERYTHING ELSE
	drop_bright, n_droplets, mask_img = findDroplets(img, cy5_img)

	#If there are no droplets, return the fact that there are no droplets and exit out
	if n_droplets == 0:
		return('no droplets', 0)
	
	####STEPS: THRESHOLD, AND GRADIENT DETECT TO FIND # OF WHOLE CELLS
	ret,thresh_img = cv2.threshold(mask_img, 40, 255, cv2.THRESH_BINARY)

	mask_img = cv2.bitwise_and(mask_img, thresh_img, thresh_img)

	####Diagnosising if Droplets have moved between imaging the brightfield and Cy5 channels
	if DIAGNOSIS_MODE:
		cv2.imshow('Masked Image', mask_img)
		cv2.waitKey()
	####End diagnosis
	
	mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
	image, contours, ____ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
	n_whole_cells = 0
	
	for c in range(len(contours)):
		area = cv2.contourArea(contours[c])
		#Rounding area for nicer output
		area = round(area, 2) 
		#most erroneous areas at this point smaller than 10 px
		if area < CELL_MAX_SIZE:
			n_whole_cells+=1
	return(drop_bright, n_whole_cells)
	
def main():
	#Create the output file
	print str(sys.argv)
	if len(sys.argv) > 1:
		if sys.argv[1] == '0':
			print('Running in Dianostic Mode')
			global DIAGNOSIS_MODE
			DIAGNOSIS_MODE = True
	
	root = os.path.abspath('.')
	outputFile = os.path.join(root, "output.csv")
	output = open(outputFile, "w+") #create/open the output folder
	output.write('Filename,Number of Intact Cells,Number of Droplets\n')

	####STEP: LOOP THROUGH LIST OF FILES (HAND GENERATED B/C THEY ARE SHORT)
	files = ['20mMNaOH', 'Clonetech', 'DropSeq', 'GeneluteLPA', 'TritonX', 'Igepal2', 'LysisEZ']
	#files = ['20mMNaOH']
	for i in range(len(files)):
		drop_brights, n_whole_cells = analyzeFile(files[i], root)
		output.write(files[i]+','+str(n_whole_cells)+ ',' + str(len(drop_brights)) + '\n')
		output.write(','.join(str(d) for d in drop_brights)+'\n')
	output.close()

	return()	

if __name__ == "__main__":
	main()