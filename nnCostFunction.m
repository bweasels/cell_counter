function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%unroll y into a matrix of logical results
y_mat = eye(num_labels)(y,:);

%compute first node forward prop
a1 = [ones(size(X,1),1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);

%compute second node forward prop 
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

%compute cost function
J = (1/m)*sum(sum(-y_mat.*log(a3)-(1-y_mat).*log(1-a3), 2));

%compute lambda penalty to large thetas
%sum sum turns it into a scalar. 
%the ,2 at the end of the first sum tells it to sum across the columns, default is rows
%We don't use theta0 so exclude theta0 from the lambda penalty
lambda_theta1 = sum(sum(Theta1(:, 2:end).^2, 2));
lambda_theta2 = sum(sum(Theta2(:, 2:end).^2, 2));


%compute the regularization expression
reg_ex = (lambda/(2*m))*(lambda_theta1 + lambda_theta2);

%add the reg_ex
J = J + reg_ex;

%Back propagate and calculate the deltas
%small delta (error) for the last node is simply node - Y training value
d3 = a3-y_mat;
%for d2 have to work backwards a bit
%equation is d2 = theta2'*d3.*g'(z2)
%inside sigmoid gradient, just adding 1s to account for the bias node
d2 = (d3*Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]));
%now remove the bias node values
d2 = d2(:,2:end);

%accumulate the gradients
%equation is Delta = Delta + d(n+1)*a(n)T
Delta_1 = d2'*a1;
Delta_2 = d3'*a2;

%Divide accumulated grads by 1/m for unregularized gradient
Theta1_grad = Delta_1./m;
Theta2_grad = Delta_2./m;

%regularizing the gradient
%zeros(size... , 1) replaces the first column values with costless 0s
reg_ex_grad1 = (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:,2:end)];
reg_ex_grad2 = (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:,2:end)];

Theta1_grad = Theta1_grad + reg_ex_grad1;
Theta2_grad = Theta2_grad + reg_ex_grad2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end