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

% Add bias
X = [ones(m, 1) X];

% output vector (Global variable)
h_theta = zeros(num_labels, 1);

% var for cost function
ss = 0;

% var for forward and backprop
delta_3  = zeros(num_labels, 1);
delta_2 = zeros(hidden_layer_size + 1, 1);
Delta_2 = zeros(num_labels, hidden_layer_size + 1);
Delta_1 = zeros(hidden_layer_size, input_layer_size + 1);

for t = 1:m
  % This line code keeps update of y_i to compute delta_3
  y_i = zeros(num_labels, 1);
  
  % ========== Forward =============
  
  % Get training example at index-th of dataset.
  x = X(t, :);
  %compute z2
  a1 = x';
  z2 = Theta1 * a1;
  
  % compute z3
  a2 = sigmoid(z2);
  a2 = [1; a2]; % add bias +1 to this layer of nodes.
  z3 = Theta2 * a2;
  
  % output
  h_theta = sigmoid(z3); % h_theta = a3
  
% ========== Backprop ============
  % Implement the cost function (No Vectorization)
  y_i(y(t)) = 1;
  for k = 1:num_labels
    % for cost function
    ss += -y_i(k) * log(h_theta(k)) - (1 - y_i(k)) * log(1 - h_theta(k));
    
    % compute delta_3
    delta_3(k) = h_theta(k) - y_i(k);
  endfor
  
  % compute delta_2
  delta_2 = (Theta2(:, 2:end)' * delta_3) .* sigmoidGradient(z2);
  
  % accumulate Delta_2
  Delta_2 += delta_3 * a2';
  
  % accumulate Delta_1
  Delta_1 += delta_2 * a1';
  
endfor

% ========= Cost without regularization =========
J = (1 / m) * ss;

% ========= Cost with regularization ==========
% Remove 1's column of each Theta vector
temp_theta1 = Theta1(:, 2:end);
temp_theta2 = Theta2(:, 2:end);
%Theta1(:, 1) = [];
%Theta2(:, 1) = [];

% Force these Theta vectors to vector form.
temp_theta1 = temp_theta1(:);
temp_theta2 = temp_theta2(:);
%Theta1 = Theta1(:);
%Theta2 = Theta2(:);

% Get sum of squares
sum_sqr_theta1 = sum(temp_theta1.^2);
sum_sqr_theta2 = sum(temp_theta2.^2);

% Get cost!
J += (lambda / (2*m)) * (sum_sqr_theta1 + sum_sqr_theta2);


% ======= compute gradient without regularization ========
Theta1_grad = (1 / m) * Delta_1;
Theta2_grad = (1 / m) * Delta_2;

% ======= compute gradient with regularization ========
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
