%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: cell_counter.m
% Purpose: Find Droplets in an image
% Notes: This is my first time!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%reshape returns the flattened image to its col and row as tested by imwrite
%image = reshape(positiveSet, col, row);

pkg load image;
%clear the environment
clear;

%Set paths to the positive and negative folders
posFolder = 'C:/Users/benku/Dropbox/Ben/Mass General Hospital Work/Mark Image Processing/positiveSet';
negFolder = 'C:/Users/benku/Dropbox/Ben/Mass General Hospital Work/Mark Image Processing/negativeSet';
imLength = 32;

[trainingSet, cvSet, testSet] = loadTrain(posFolder, negFolder, imLength);

Y = trainingSet(:,1);
trainingSet = trainingSet(:, 2:end);

Y_cv = cvSet(:, 1);
cvSet = cvSet(:, 2:end);

Y_test = testSet(:, 1);
testSet = testSet(:, 2:end);

initial_theta = zedros(size(trainingSet, 2), 1);
iterations = [500, 600, 700, 800, 1000];
lambdas = 1.2;
for i = 1:length(iterations)

  options = optimset('GradObj', 'on', 'MaxIter', iterations(i));

  fprintf('starting fminunc on lambda: %d\n', lambdas(i))
  [theta, cost, output] = fminunc(@(t) costFunctionReg(t, trainingSet, Y, lambdas(i)), initial_theta, options);

  fprintf('Training Set Cost (lambda %d) is : %d\n', lambdas, cost)
  fprintf('fminunc returned with code %d\n', output)

  predictions = predict(theta, cvSet);
  
  [TruePos, TrueNeg, FalsePos, FalseNeg] = confusionMatrix(Y_cv, predictions);

  %Compare the output with the predictions and only place 1s where they don't match
  %sum those, divide by the length and multiply by 100 
  perIncorrect = sum(Y_cv != predict(theta, cvSet))/length(Y_cv)*100;
  fprintf('Confusion Matrix\n        --------\nTrue Pos  |%d|%d| False Neg\n', TruePos, FalseNeg)
  fprintf('False Pos |%d|%d| True Neg\n        --------\n', FalsePos, TrueNeg)
  fprintf('The percent of the Cross Validation set that is missclassified: %d\n', perIncorrect)
  fprintf('-------------------------------------------\n')

end
%%%%%
%TO DO: Try various dimensions to get the lowest CV:
%TO DO: Try various lambdas to get the lowest CV: surprisingly 1.2 was a minimum
%TO DO: Plot Jtrain and JCV against number of samples (M):  


