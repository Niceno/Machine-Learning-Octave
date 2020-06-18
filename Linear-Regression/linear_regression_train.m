%===============================================================================
  function [theta mu sigma x_normalized J_history] = ...
    linear_regression_train(x, y, alpha, lambda, numb_i)
%-------------------------------------------------------------------------------
% x      - training set.
% y      - training set output values.
% alpha  - learning rate (gradient descent step size).
% lambda - regularization parameter.
% numb_i - number of gradient descent steps.
%-------------------------------------------------------------------------------

  % Calculate the number of training examples (1st dimension is number of rows)
  m = size(y, 1);

  % Calculate the number of features (2nd dimension is number of columns)
  n = size(x, 2);

  % Normalize features
  [x_normalized mu sigma] = feature_normalize(x);

  % Add a column of ones to x
  x_normalized = [ones(m, 1), x_normalized];

  % Initialize model parameters
  initial_theta = zeros(n + 1, 1);

  % Run gradient descent
  [theta, J_history] = gradient_descent(x_normalized,    ...
                                        y,               ...
                                        initial_theta,   ...
                                        alpha,           ...
                                        lambda,          ...
                                        numb_i);

end
