import cv2
import os
import gc
import numpy

CELL_MAX_SIZE = 20

def findDroplets(img, cy5_img):
	
	height, width = img.shape
	
	#threshold for houghCircles
	ret,thresh_img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
	
	#houghCircles params
	circles = cv2.HoughCircles(thresh_img, cv2.HOUGH_GRADIENT, dp=2, minDist=40, param1=40, param2=40, minRadius=25, maxRadius=60)
	
	#convert circles to unsigned 16bit ints so can be used to draw new circles
	circles = numpy.uint16(numpy.around(circles))
	
	#initialize Masking image
	mask_img = numpy.zeros((height, width, 3), numpy.uint8)
	
	#draw circles on the mask
	for i in circles[0,:]:
		cv2.circle(mask_img, (i[0],i[1]),i[2],(255,255,255),-1)
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
	#mask_img = cv2.erode(mask_img, kernel, iterations = 1)
	
	#output number of circles (again QC shit)
	n_droplets = len(circles[0])

	#mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
	blanked = cv2.bitwise_and(cy5_img, mask_img, mask_img)
	
	return(n_droplets, blanked)
	
def dropPixelVal(img):
	height, width = img.shape
	grossPixelVal = 0
	count = 0
	
	##Go through and add up pixel values. If the pixel value is 0 then it has been masked out and can be ignored
	for h in range(height):
		for w in range(width):
			if img[h][w] != 0:
				grossPixelVal += img[h][w]
				count+=1
	#grossPixelVal = grossPixelVal/count
	return(grossPixelVal) 

def analyzeFolder(filename, root):	
	###get the image path and load them
	BFaddr = filename+'_BF.jpg'
	CY5addr = filename+'_Cy5.jpg'
	print('Opening files: ' + BFaddr + ', ' + CY5addr)
	
	imgAddr = os.path.join(root, BFaddr)
	cy5imgAddr = os.path.join(root, CY5addr)
	img = cv2.imread(imgAddr, 0)
	cy5_img = cv2.imread(cy5imgAddr, 1)
	
	####STEP: FIND DROPLETS BY MASKING EVERYTHING ELSE
	n_droplets, mask_img = findDroplets(img, cy5_img)
	mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
	
	####STEP: GET AVG PIXEL VALUE OF ENCAPSULATED DROPLETS
	avgPixel = dropPixelVal(mask_img)
	avgPixelperDroplet = avgPixel/n_droplets
	
	####STEPS: THRESHOLD, AND GRADIENT DETECT TO FIND # OF WHOLE CELLS
	ret,thresh_img = cv2.threshold(mask_img, 30, 255, cv2.THRESH_BINARY)
	mask_img = cv2.bitwise_and(mask_img, thresh_img, thresh_img)
	image, contours, __ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
	n_whole_cells = 0
	
	for c in range(len(contours)):
		area = cv2.contourArea(contours[c])
		#Rounding area for nicer output
		area = round(area, 2) 
		#Area multiplied by scale^2 because units are pixels^2
		#most erroneous areas at this point smaller than 10 px
		if area < CELL_MAX_SIZE:
			n_whole_cells+=1
	return(avgPixelperDroplet, n_whole_cells)
	
def main():

	#Create the output file
	root = os.path.abspath('.')
	outputFile = os.path.join(root, "output.csv")
	output = open(outputFile, "w+") #create/open the output folder
	output.write('Filename,Average Glow Per Droplet,Number of Intact Cells\n')

	####STEP: LOOP THROUGH LIST OF FILES (HAND GENERATED B/C THEY ARE SHORT)
	files = ['20mMNaOH', 'Clonetech', 'Clonetech2', 'Dropseq', 'Dropseq2', 'GeneluteLPA', 'RTBuffer', 'TritonX']
	for i in range(len(files)):
		(avgPixelperDroplet, n_whole_cells) = analyzeFolder(files[i], root)
		output.write(files[i]+','+str(avgPixelperDroplet)+','+str(n_whole_cells)+'\n')
	output.close()
	return()	

if __name__ == "__main__":
	main()