%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: loadTrain.m
% Purpose: Loads the training Set
% Notes: This is my first time!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [trainingSet, cvSet, testSet] = loadTrain(positiveFolder, negativeFolder, nFeatures, nTrain = 0)

%size of training Set and CV and Test
TrainEnd=0.7;
CVend = 0.85;

%define the positive, negative, and output variables
positiveSet = zeros(nFeatures^2,1);
negativeSet = zeros(nFeatures^2,1);

positiveCV = zeros(nFeatures^2,1);
negativeCV = zeros(nFeatures^2,1);

positiveTest = zeros(nFeatures^2,1);
negativeTest = zeros(nFeatures^2,1);

Y = 0;
Y_CV = 0;
Y_Test = 0;

%set the directory to positive folder and negative folder
posFiles = dir(positiveFolder);
negFiles = dir(negativeFolder);

%define the size of trainingSet data to load
if (nTrain == 0)
  nPos = length(posFiles);
  nNeg = length(negFiles);
elseif (nTrain > length(posFiles))
  nPos = length(posFiles);
  nNeg = length(negFiles);
  fprintf('Input Size is larger than number of training examples allowed')
else
  nPos = nTrain;
  nNeg = nTrain;
endif

%go through and get positive images
for i = 3:int32(nPos*TrainEnd)
   imgAddr = strcat('positiveSet/', posFiles(i).name);
   flat = loadImg(imgAddr);
   positiveSet = [positiveSet, flat];
   Y = [Y;2];
end

%fprintf('loaded Postive Set\n') diagnostics

for i = int32(nPos*TrainEnd):int32(nPos*CVend)
   imgAddr = strcat('positiveSet/', posFiles(i).name);
   flat = loadImg(imgAddr);
   positiveCV = [positiveCV, flat];
   Y_CV = [Y_CV;2];
end

%fprintf('loaded Positive Cross Validation Set\n')

for i = int32(nPos*CVend):nPos
   imgAddr = strcat('positiveSet/', posFiles(i).name);
   flat = loadImg(imgAddr);
   positiveTest = [positiveTest, flat];
   Y_Test = [Y_Test;2];
end

%fprintf('loaded Postive Test Set\n')

%remove the first rows of zeros used to define the positive set and rotate
positiveSet(:,[1]) = [];
positiveCV(:,[1]) = [];
positiveTest(:,[1]) = [];

%transpose to get everything organized properly
positiveSet = positiveSet';
positiveCV = positiveCV';
positiveTest = positiveTest';

for j = 3:int32(nNeg*0.8)
   imgAddr = strcat('negativeSet/', negFiles(j).name);
   flat = loadImg(imgAddr);
   negativeSet = [negativeSet, flat];
   Y = [Y;1];
end

%fprintf('loaded Negative Set\n')

for i = int32(nNeg*0.8):int32(nNeg*0.9)
   imgAddr = strcat('negativeSet/', negFiles(i).name);
   flat = loadImg(imgAddr);
   negativeCV = [negativeCV, flat];
   Y_CV = [Y_CV;1];
end

%fprintf('loaded Negative Cross Validation Set\n')

for i = int32(nNeg*0.9):nNeg
   imgAddr = strcat('negativeSet/', negFiles(i).name);
   flat = loadImg(imgAddr);
   negativeTest = [negativeTest, flat];
   Y_Test = [Y_Test;1];
end

%fprintf('loaded Negative Test Set\n')

%remove first row of zeros
negativeSet(:,[1]) = [];
negativeCV(:,[1]) = [];
negativeTest(:,[1]) = [];

%transpose
negativeSet = negativeSet';
negativeCV = negativeCV';
negativeTest = negativeTest';

%concatenate the positive and negative set to get the training set
trainingSet = [positiveSet;negativeSet];
cvSet = [positiveCV; negativeCV];
testSet = [positiveTest; negativeTest];

%Remove he first y to get rid of the nul value
Y(1) = [];
Y_CV(1) = [];
Y_Test(1) = [];

%Make training set output
[rows,_] = size(trainingSet);
intercept = ones(rows, 1);
trainingSet = [intercept trainingSet];
trainingSet = [Y trainingSet];
%fprintf('Finished up training set\n')

%Make CV output
[row_cv,_] = size(cvSet);
intercept = ones(row_cv,1);
cvSet = [intercept cvSet];
cvSet = [Y_CV cvSet];
%fprintf('Finished up CV Set\n')

%Make test set output
[row_test,_] = size(testSet);
intercept = ones(row_test,1);
testSet = [intercept testSet];
testSet = [Y_Test testSet];
%fprintf('Finished up test Set\n')

endfunction