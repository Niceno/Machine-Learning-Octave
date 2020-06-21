%===============================================================================
function [predictions] = hypothesis(x, theta)
%-------------------------------------------------------------------------------
% Predicts the output values y based on the input values x and model parameters.
%-------------------------------------------------------------------------------
% Input:
% x     - input features       - (m x n) matrix.
% theta - our model parameters - (n x 1) vector.
%
% Output:
% predictions - output values based on model parameters - (m x 1) vector.
%
% Where:
% m - number of training examples,
% n - number of features.
%-------------------------------------------------------------------------------

  predictions = x * theta;

  end
