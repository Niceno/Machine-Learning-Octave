%===============================================================================
  function [theta, j_history] =  ...
    gradient_descent(x, y, theta, alpha, lambda, numb_i)
%-------------------------------------------------------------------------------
% Calculates what steps (deltas) should be taken for each
% theta parameter in order to minimize the cost function.
%-------------------------------------------------------------------------------
% Input:
% x      - training set of features - (m x n) matrix.
% y      - a vector of expected output values - (m x 1) vector.
% theta  - current model parameters - (n x 1) vector.
% alpha  - learning rate, the size of gradient step at each iteration.
% lambda - regularization parameter.
% numb_i - number of iterations we will take for gradient descent.
%
% Output:
% theta - optimized theta parameters - (m x 1) vector.
% j_history - the history cost function changes over iterations.
%
% Where:
% m  - number of training examples,
% n - number of features.
%-------------------------------------------------------------------------------

  % Get number of training examples.
  m = size(x, 1);

  % Initialize j_history with zeros.
  j_history = zeros(numb_i, 1);

  for iteration = 1:numb_i
    % Perform a single gradient step on the parameter vector theta.
    theta = gradient_step(x, y, theta, alpha, lambda);

    % Save the cost J in every iteration
    j_history(iteration) = cost_function(x, y, theta, lambda);
  end

  end
