%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: predict.m
% Purpose: Uses Theta to produce predictions
% Notes: This is my first time!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function predictions = predict(theta, X)

%initialize variables
nItems = size(X, 1);
predictions = zeros(nItems, 1);

%Apply theta to the inputs
predictions = sigmoid(X*theta);

#Turn them into 1s and 0s 
predictions(find(predictions >= 0.5)) = 1;
predictions(find(predictions < 0.5)) = 0;

endfunction