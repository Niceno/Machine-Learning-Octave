%===============================================================================
  function [x_normalized, mu, sigma] = ...
    feature_normalize(x)
%-------------------------------------------------------------------------------
% Normalizes the features in x. Returns a normalized version of x where
% the mean value of each feature is 0 and the standard deviation is 1.
%-------------------------------------------------------------------------------

  % Calculate the number of features (number of columns in x)
  n = size(x, 2);

  % Initialize x_normalized to be the same as x
  x_normalized = x;

  % Initialize average value and standard deviation
  mu    = zeros(1, n);
  sigma = zeros(1, n);

  % Get average values for each feature (column) in x.
  % (Function "mean(A)" returns the mean of the elements of A
  %  along the first array dimension whose size does not equal 1.
  %  In this case, first dimension is through rows, hence "mean"
  %  returns the average in each column.)
  mu = mean(x_normalized);

  % Calculate the standard deviation for each feature.
  % (Function "std(A)" returns the standard deviation of the elements
  %  of A along the first array dimension whose size does not equal 1.
  %  In this case, first dimension is through rows, hence "std"
  %  returns the standard deviation in each column.)
  sigma = std(x_normalized);

  % Subtract mean values from each feature (column) of every example (row)
  % to make all features be spread around zero.
  x_normalized = x_normalized - mu;

  % Normalize each feature values for each example so that all features 
  % are close to [-1:1] boundaries.
  x_normalized = x_normalized ./ sigma;

end
