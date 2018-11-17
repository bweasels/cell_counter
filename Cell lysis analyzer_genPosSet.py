import cv2
import os
import gc
import sys
import numpy

CELL_MAX_SIZE = 20
DROP_RADIUS = 32

DIAGNOSIS_MODE = True


#####SUPPORT FUNCTIONS FOR FIND DROPLET#####
def cropImg(img, y, x, height, width, radius):
	#Add or subtract the droplet radius 
	Top = y-radius
	Bottom = y+radius
	Left = x-radius
	Right = x+radius
	
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

#####FINDS DROPLETS AND QUANTS THE PIXEL VALUES IN EACH DROPLET#####
def findDroplets(img):
	height, width = img.shape
	
	#threshold for houghCircles
	ret,thresh_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
	
	#houghCircles params
	circles = cv2.HoughCircles(thresh_img, cv2.HOUGH_GRADIENT, dp=2, minDist=40, param1=70, param2=50, minRadius=25, maxRadius=40)
	
	####Generating Positive Set####
	temp_img = img
	temp_circles = circles
	if temp_circles is not None:
		temp_circles = numpy.round(temp_circles[0, :]).astype("int")
		for (x, y, r) in temp_circles:
			crop_img = cropImg(temp_img, y, x, 1040, 1392, r)
			address = os.path.join(os.getcwd(),('positiveSet/'+str(x)+str(y)+'.jpg'))
			cv2.imwrite(address, crop_img)
	####End Positive Set Generation####
	
	return()

#####WRAPPER FUNCTION THAT ANALYZES THE GIVEN FILES#####
def analyzeFile(filename, root):	
	#get the image path
	BFaddr = filename+'_BF.jpg'
	print('Opening files: ' + BFaddr)
	
	#Load images
	imgAddr = os.path.join(root, BFaddr)
	img = cv2.imread(imgAddr, 0)
	
	####STEP: FIND DROPLETS BY MASKING EVERYTHING ELSE
	findDroplets(img)

	return()
	
def main():
	#Create the output file
	print(str(sys.argv))

	
	root = os.path.abspath('.')
	outputFile = os.path.join(root, "output.csv")
	output = open(outputFile, "w+") #create/open the output folder
	output.write('Filename,Number of Intact Cells,Number of droplets\n')

	####STEP: LOOP THROUGH LIST OF FILES (HAND GENERATED B/C THEY ARE SHORT)
	files = ['20mMNaOH', 'Clonetech', 'DropSeq', 'GenEluteLPA', 'TritonX', 'Igepal', 'LysisEZ', 'RTBuffer']
	#files = ['20mMNaOH']
	for i in range(len(files)):
		analyzeFile(files[i], root)
	output.close()

	return()	

if __name__ == "__main__":
	main()