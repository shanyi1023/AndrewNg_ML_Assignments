function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the prediction for that example. You can make use of this to vectorize the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, there're many possible vectorized solutions, but one solution looks like:
%     grad = (unregularized gradient for logistic regression)
%     temp = theta; 
%     temp(1) = 0;   % because we don't add anything for j = 0  
%     grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% ===================Solution 1================================

%regularized cost function
% using vector multiplication. The size of each argument is (m x 1), and we want the vector product to be a scalar, 
% so use a transposition so that (1 x m) times (m x 1) gives a result of (1 x 1), a scalar

h = sigmoid(X*theta);
J = ((-y')*log(h)-(1-y)'*log(1-h)) / m + lambda*sum(theta(2:end).^2) / (2*m)

%regularized gradient descent
theta(1) = 0
grad = (X'*(h - y) + lambda*theta)/m;

% ===================Solution 2================================

% calculate cost function
% h = sigmoid(X*theta);

% calculate penalty
% excluded the first theta value, we want the regularization to exclude the bias feature, so we can set theta(1) to zero or change the whole vector as below
% theta1 = [0 ; theta(2:size(theta), :)];
% p = lambda*(theta1'*theta1)/(2*m);
% J = ((-y)'*log(h) - (1-y)'*log(1-h))/m + p;

% calculate grads
% grad = (X'*(h - y)+lambda*theta1)/m;

% =============================================================

grad = grad(:);

end
