function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% error vector of all training examples
error = X * theta - y;

sqr_error = error.^2; % Other, error.*error
sum_sqr_error = sum(sqr_error);

J = (1 / (2*m)) * sum_sqr_error;


% =========================================================================

end
