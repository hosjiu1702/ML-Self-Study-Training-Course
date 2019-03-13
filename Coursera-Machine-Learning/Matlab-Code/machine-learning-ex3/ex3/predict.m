function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add bias node
X = [ones(size(X, 1), 1) X];

for j = 1:m
  % Get training example at j-th of X dataset
  x = X(j, :);
  % compute z2
  a1 = x';
  z2 = Theta1 * a1;
  % compute a2
  a2 = sigmoid(z2);
  
  % compute z3
  a2 = [1; a2];
  z3 = Theta2 * a2;
  % compute a3
  a3 = sigmoid(z3);
  
  % compute output
  h = a3;
  
  % Get element and its index has highest probability in output vector above
  [max_value max_index] = max(h)
  
  % convert to p
  p(j) = max_index;
  
endfor

return




% =========================================================================


end
