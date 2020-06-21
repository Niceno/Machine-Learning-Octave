%===============================================================================
% Demonstrate logistic regression
%-------------------------------------------------------------------------------

% Clear variables and a screen.
clear; close all; clc;

POLY_DEGREE = 2;
LAMBDA      = 1.0e-9;

%----------------------------------------
%
% Define a non-linear scattered data set
% x - features
% y - results
%
%----------------------------------------

%-----------------
% Define features
%-----------------
load bubble_05.dat
bubble_05 = flip(bubble_05);

R = size(bubble_05);  % resolution of the input data

x = [meshgrid(linspace(-1, 1, R(2)), linspace(-1,1,R(1)))(:)    ...
     meshgrid(linspace(-1, 1, R(1)), linspace(-1,1,R(2)))'(:)]

y = bubble_05(:);

%--------------------
% Plot training data
%--------------------
fprintf('Plotting the data...\n\n');

% Find indices of ones and twos, discard zeroes
ind_1 = find(y == 1);
ind_2 = find(y == 2);

% Plot examples.
hold on;
axis equal;
plot(x(ind_1, 1), x(ind_1, 2), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
plot(x(ind_2, 1), x(ind_2, 2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 8);

x = x([ind_1; ind_2], :);
y = y([ind_1; ind_2], :);
y = y .- 1;

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
polynomial_degree = POLY_DEGREE;
x = add_polynomial_features(x(:, 1), x(:, 2), polynomial_degree);
size(x)

% Run the regression.
lambda = LAMBDA;
[theta, J, J_history, exit_flag] = ...
  logistic_regression_train(x, y, lambda);

fprintf('- Initial cost: %f\n', J_history(1));
fprintf('- Optimized cost: %f\n\n', J);

%------------------------------
% Plotting decision boundaries
%------------------------------
fprintf('Plotting decision boundaries...\n\n');

% Generate a grid range.
G = 64;
u = linspace(-1, 1, G);
v = linspace(-1, 1, G);
z = zeros(length(u), length(v));

% Evaluate z = (x * theta) over the grid
for i = 1 : G
  for j = 1 : G

    % Add polinomials
    x = add_polynomial_features(u(i), v(j), polynomial_degree);

    % Add ones
    x = [ones(size(x, 1), 1), x];
    z(i, j) = x * theta;
  end
end

% Plot z = 0
% Notice you need to specify the range [0, 0]
contour(u, v, z', [0, 0], 'LineWidth', 4);      % send transposed z
title(sprintf('lambda = %g \n order = %g', LAMBDA, POLY_DEGREE));
legend('y = 1', 'y = 0', 'Bubble surface');

hold off;

