%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: mlalgo.m
% Purpose: Find Droplets in an image
% Notes: This is my first time!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%reshape returns the flattened image to its col and row as tested by imwrite
%image = reshape(positiveSet, col, row);
%% Initialization
clear ; close all; clc
pkg load image;

%Set paths to the positive and negative folders
posFolder = 'C:/Users/benku/cell_counter/positiveSet';
negFolder = 'C:/Users/benku/cell_counter/negativeSet';
imLength = 24;

[trainingSet, cvSet, testSet] = loadTrain(posFolder, negFolder, imLength, 500);

Y = trainingSet(:,1);
trainingSet = trainingSet(:, 2:end);

Y_cv = cvSet(:, 1);
cvSet = cvSet(:, 2:end);

Y_test = testSet(:, 1);
testSet = testSet(:, 2:end);

initial_theta = zeros(size(trainingSet, 2), 1);

imgSize = size(trainingSet, 2);


##############START NN CODE#####################
input_layer_size = imgSize;
hidden_layer_size = imgSize;
num_labels = 2;
lambda = 0;

fprintf('\nTraining Neural Network on lambda: %d\n', lambda)

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 100);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, trainingSet, Y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));  

predict = predictNN(Theta1, Theta2, cvSet);
per_Incorrect = sum(Y_cv != (predict))/length(Y_cv)                 
%[TruePos, TrueNeg, FalsePos, FalseNeg] = confusionMatrix(Y_cv, predictions);

  %Compare the output with the predictions and only place 1s where they don't match
  %sum those, divide by the length and multiply by 100 
#perIncorrect = sum(Y_cv != predict(theta, cvSet))/length(Y_cv)*100;
#fprintf('Confusion Matrix\n        --------\nTrue Pos  |%d|%d| False Neg\n', TruePos, FalseNeg)
#fprintf('False Pos |%d|%d| True Neg\n        --------\n', FalsePos, TrueNeg)
#fprintf('The percent of the Cross Validation set that is missclassified: %d\n', perIncorrect)
#fprintf('-------------------------------------------\n')

%%%%%
%TO DO: Try various dimensions to get the lowest CV:
%TO DO: Try various lambdas to get the lowest CV: surprisingly 1.2 was a minimum
%TO DO: Plot Jtrain and JCV against number of samples (M):  


