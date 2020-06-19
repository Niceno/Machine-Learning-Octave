%===============================================================================
% Demonstrate linear regression
%-------------------------------------------------------------------------------

% Clear variables and a screen.
clear; close all; clc;

%---------------------------------
% Loading training data from file
%
% File looks like this:
% 2104,3,399900
% 1600,3,329900
% 2400,3,369000
% 1416,2,232000
% 3000,4,539900
% 1985,4,299900
% ...
%---------------------------------
fprintf('Loading the training data from file...\n\n');

data = load('house_prices.csv');

%----------------------------------------------
% Split data into features (x) and results (y)
%----------------------------------------------
x = data(:, 1: 2);  % size in square feet and number of rooms
y = data(:, 3);     % price

%------------------------
% Plotting training data
%------------------------
fprintf('Plotting the training data...\n\n');

% Split the figure on 2x2 sectors and start drawing in first sector.
subplot(2, 2, 1);

scatter3(x(:, 1), x(:, 2), y, [], y(:), 'o');
title('Training Set');
xlabel('Size');
ylabel('Rooms');
zlabel('Price');

%---------------------------
% Running linear regression
%---------------------------
fprintf('Running linear regression...\n');

% Setup regularization parameter.
lambda =  0;
alpha  =  0.1;
num_i  = 50;
[theta mu sigma x_normalized J_history] = ...
  regression_train(x,                     ...
                   y,                     ...
                   alpha,                 ...
                   lambda,                ...
                   num_i);

fprintf('- Initial cost: %f\n', J_history(1));
fprintf('- Optimized cost: %f\n', J_history(end));

fprintf('- Theta (with normalization):\n');
fprintf('-- %f\n', theta);
fprintf('\n');

%--------------------------------------------------
% Calculate model parameters using normal equation
%--------------------------------------------------
fprintf('Calculate model parameters using normal equation...\n');

x_normal     = [ones(size(x, 1), 1) x];
theta_normal = normal_equation(x_normal, y);
normal_cost  = cost_function(x_normal, y, theta_normal, lambda);

fprintf('- Normal function cost: %f\n', normal_cost);

fprintf('- Theta (without normalization):\n');
fprintf('-- %f\n', theta_normal);
fprintf('\n');

%-----------------------------------
% Plotting normalized training data
%-----------------------------------
fprintf('Plotting normalized training data...\n\n');

% Start drawing in second sector.
subplot(2, 2, 2);

scatter3(x_normalized(:, 1), x_normalized(:, 2), y, [], y(:), 'o');
title('Normalized Training Set');
xlabel('Normalized Size');
ylabel('Normalized Rooms');
zlabel('Price');

%--------------------------------
% Draw gradient descent progress
%--------------------------------
fprintf('Plot gradient descent progress...\n\n');

% Continue plotting to the right area.
subplot(2, 2, 3);

plot(1:num_i, J_history);
xlabel('Iteration');
ylabel('J(\theta)');
title('Gradient Descent Progress');

%--------------------------------------------------
% Plotting hypothesis plane on top of training set
%--------------------------------------------------
fprintf('Plotting hypothesis plane on top of training set...\n\n');

% Get apartment size and rooms boundaries.
x1  = x_normalized(:, 1);
x2  = x_normalized(:, 2);
xn1 = linspace(min(x1), max(x1), 10);
xn2 = linspace(min(x2), max(x2), 10);

% Calculate predictions for each possible combi-
% nation of rooms number and appartment size.
y_h = zeros(length(xn1), length(xn2));
for i1 = 1:length(xn1)
  for i2 = 1:length(xn2)
      x = [1, xn1(i1), xn2(i2)];
      y_h(i1, i2) = hypothesis(x, theta);
  end
end

% Plot the plane on top of training data to see how it feets them.
subplot(2, 2, 2);
hold on;
mesh(xn1, xn2, y_h);
legend('Training Examples', 'Hypothesis Plane')
hold off;
