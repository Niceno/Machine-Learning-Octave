%===============================================================================
function [theta, J, j_history, exit_flag] = ...
  logistic_regression_train(x, y, lambda)
%-------------------------------------------------------------------------------
% x      - training set.
% y      - training set output values.
% lambda - regularization parameter.
%-------------------------------------------------------------------------------

  % Calculate the number of training examples.
  m = size(y, 1);

  % Calculate the number of features.
  n = size(x, 2);

  % Add a column of ones to x.
  x = [ones(m, 1), x];

  % Initialize model parameters.
  initial_theta = zeros(n + 1, 1);

  % Run gradient descent.
  [theta, J, exit_flag] = gradient_descent(x, y, initial_theta, lambda);

  % Record the history of chaning J.
  j_history = zeros(1, 1);
  j_history(1) = cost_function(x, y, initial_theta, lambda);
  j_history(2) = cost_function(x, y, theta, lambda);

  end
