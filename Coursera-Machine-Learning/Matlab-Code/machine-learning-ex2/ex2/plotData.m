function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

m = length(y);

% Create New Figure
figure;
hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos_point = [NaN, NaN];
neg_point = [NaN, NaN];

% Separating neg point and pos point into 2 separated vector
for i = 1:m
  if (y(i) == 0)
    neg_point = [neg_point; X(i, 1), X(i, 2)];
  else
    pos_point = [pos_point; X(i, 1), X(i, 2)];
  endif
endfor

% Remove [NaN NaN] from two those vector
neg_point(1, :) = [NaN, NaN];
pos_point(1, :) = [NaN, NaN];

plot(neg_point(:, 1), neg_point(:, 2), 'or', 'markersize', 10, 'markerfacecolor', 'r');
plot(pos_point(:, 1), pos_point(:, 2), '+k', 'markersize', 10, 'linewidth', 2);

% Put some labels and legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;



% =========================================================================

end
