%===============================================================================
% Demonstrate logistic regression
%-------------------------------------------------------------------------------

% Clear variables and a screen.
clear; close all; clc;

%----------------------------------------
%
% Define a non-linear scattered data set
% x - features
% y - results
%
%----------------------------------------
m = 512;
n =   2;

%-----------------
% Define features
%-----------------

% Define data ranges
x_min = [-1, -1];
x_max = [+1, +1];

% Assign values in given data range ...
x = zeros(m, n);
for j = 1 : n
  x(:,j) = [x_min(j) : (x_max(j)-x_min(j))/(m-1) : x_max(j)]';
end

% ... and shuffle them to make it more interesting
for j = 1 : n
  x(:,j) = x( randperm( size(x,1) ), j);
end

y = zeros(m,1);
dist = sqrt((0.66*x(:,1)).^2 .+ x(:,2).^2);
y = dist < 0.5;

%--------------------
% Plot training data
%--------------------
fprintf('Plotting the data...\n\n');

% Find indices of positive and negative examples.
positiveIndices = find(y == 1);
negativeIndices = find(y == 0);

% Plot examples.
hold on;
plot(x(positiveIndices, 1), ...
     x(positiveIndices, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(x(negativeIndices, 1), ...
     x(negativeIndices, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% Draw labels and Legend
xlabel('x coordinate');
ylabel('y coordinate');
legend('y = 1', 'y = 0');

%-----------------------------
% Running logistic regression
%-----------------------------
fprintf('Running logistic regression...\n\n');

% Add more polynomial features in order to make
% decision boundary to have more complex curve form.
polynomial_degree = 2;
x = add_polynomial_features(x(:, 1), x(:, 2), polynomial_degree);
size(x)

% Run the regression.
lambda = 0.1;
[theta, J, J_history, exit_flag] = ...
  logistic_regression_train(x, y, lambda);

fprintf('- Initial cost: %f\n', J_history(1));
fprintf('- Optimized cost: %f\n\n', J);

%------------------------------
% Plotting decision boundaries
%------------------------------
fprintf('Plotting decision boundaries...\n\n');

% Generate a grid range.
M = 64;
u = linspace(-1, 1, M);
v = linspace(-1, 1, M);
z = zeros(length(u), length(v));

% Evaluate z = (x * theta) over the grid
for i = 1 : M
  for j = 1 : M

    % Add polinomials
    x = add_polynomial_features(u(i), v(j), polynomial_degree);

    % Add ones
    x = [ones(size(x, 1), 1), x];
    z(i, j) = x * theta;
  end
end

% Plot z = 0
% Notice you need to specify the range [0, 0]
contour(u, v, z', [0, 0], 'LineWidth', 2);      % send transposed z
title(sprintf('lambda = %g', lambda));
legend('y = 1', 'y = 0', 'Bubble surface');

hold off;

