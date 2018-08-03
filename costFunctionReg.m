%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: costFunctionReg.m
% Purpose: computes the cost of theta, and the gradient
% Notes: This is my first time!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [J, grad] = costFunctionReg(theta, X, y, lambda)

%Constants
m = length(y);
n = length(theta);

h = sigmoid(X*theta);

%regularization terminal_size
short_theta = theta(2:n);
sum_thetas = sum(short_theta.^2);

%Cost Function J

J = (1/m)*(-y'*log(h)-(1-y)'*log(1-h))+(lambda/(2*m))*(sum_thetas);

%Gradient for regularization & the gradient for the bias node
grad = (1/m)*X'*(h-y)+(lambda/m)*theta;
temp = (1/m)*X'*(h-y);

grad(1) = temp(1);

endfunction