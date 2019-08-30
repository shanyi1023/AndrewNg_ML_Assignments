function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 


% One is to use the bsxfun() function, with the @power operator,
% where vector1 is a column vector of the feature values 'X',
% and vector2 is a row vector of exponents from 1 to 'p'.
% X_poly = bsxfun(@power, vector1, vector2)

X_poly = bsxfun(@power, X, [1:p])


% Use the element-wise exponent operator '.^', and converting
% both X and the vector of exponent values into equal-sized matrices
% by multiplying each by a vectors of all-ones.

% for i=1:p
%   X_poly(:,i) = X.^i;
% end;

% =========================================================================

end
