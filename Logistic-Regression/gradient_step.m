%===============================================================================
  function [gradients] = ...
    gradient_step(x, y, theta, lambda)
%-------------------------------------------------------------------------------
% Performs one step of gradient descent for theta parameters.
%-------------------------------------------------------------------------------
% x - training set.
% y - training output values.
% theta - model parameters.
% lambda - regularization parameter.
%-------------------------------------------------------------------------------

  % Initialize number of training examples
  m = length(y);

  % Initialize variables we need to return.
  gradients = zeros(size(theta));

  % Calculate hypothesis.
  predictions = hypothesis(x, theta);

  % Calculate regularization parameter
  regularization_param = (lambda / m) * theta;

  % Calculate gradient steps
  gradients = (1 / m) * (x' * (predictions - y)) + regularization_param;

  % We should NOT regularize the parameter theta_zero
  gradients(1) = (1 / m) * (x(:, 1)' * (predictions - y));

  end
