%===============================================================================
  function [theta, j, exit_flag] = ...
    gradient_descent(x, y, theta, lambda)
%-------------------------------------------------------------------------------
% Iteratively optimizes theta model parameters.
% x - training set.
% y - training output values.
% theta - model parameters.
% lambda - regularization parameter.
%-------------------------------------------------------------------------------

  % Set Options
  options = optimset('GradObj', 'on', 'MaxIter', 1024);

  % Optimize
  [theta, j, exit_flag] =                              ...
     fminunc(@(t)(gradient_callback(x, y, t, lambda)), ...
             theta,                                    ...
             options);

end
