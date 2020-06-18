%===============================================================================
  function [theta] = ...
    gradient_step(x, y, theta, alpha, lambda)
%-------------------------------------------------------------------------------
% Function performs one step of gradient descent for theta parameters.
%-------------------------------------------------------------------------------
% Input:
% x      - training set of features - (m x n) matrix.
% y      - a vector of expected output values - (m x 1) vector.
% theta  - current model parameters - (n x 1) vector.
% alpha  - learning rate, the size of gradient step at each iteration.
% lambda - regularization parameter.
%
% Output:
% theta     - optimized theta parameters - (m x 1) vector.
% J_history - the history cost function changes over iterations.
%
% Where:
% m - number of training examples,
% n - number of features.
%-------------------------------------------------------------------------------

  % Get number of training examples.
  m = size(x, 1);

  % Predictions of hypothesis on all m examples.
  predictions = hypothesis(x, theta);

  % The difference between predictions and actual values for all m examples.
  difference = predictions - y;

  % Calculate regularization parameter.
  regularization_param = 1 - alpha * lambda / m;

  % Vectorized version of gradient descent.
  theta = theta * regularization_param - alpha * (1 / m) * (difference' * x)';

  % We should NOT regularize the parameter theta_zero.
  theta(1) = theta(1) - alpha * (1 / m) * (x(:, 1)' * difference)';

end
