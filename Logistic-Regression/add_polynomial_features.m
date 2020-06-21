%===============================================================================
  function out = add_polynomial_features(x1, x2, degree)
%-------------------------------------------------------------------------------
% Generates polinomyal features of certain degree.  This function is used to
% extend training set features with new features to get more complex shape of
% decision boundaries.
%-------------------------------------------------------------------------------

  out = ones(size(x1(:, 1)));
  for i = 1:degree
    for j = 0:i
      out(:, end + 1) = (x1 .^ (i - j)) .* (x2 .^ j);
    end
  end

  end
