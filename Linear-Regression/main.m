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

scatter3(x_normalized(:, 2), x_normalized(:, 3), y, [], y(:), 'o');
title('Normalized Training Set');
xlabel('Normalized Size');
ylabel('Normalized Rooms');
zlabel('Price');

% Draw gradient descent progress ------------------------------------------------
fprintf('Plot gradient descent progress...\n\n');

% Continue plotting to the right area.
subplot(2, 2, 3);

plot(1:num_i, J_history);
xlabel('Iteration');
ylabel('J(\theta)');
title('Gradient Descent Progress');

% Plotting hypothesis plane on top of training set -----------------------------
fprintf('Plotting hypothesis plane on top of training set...\n\n');

% Get apartment size and rooms boundaries.
apt_sizes = x_normalized(:, 2);
apt_rooms = x_normalized(:, 3);
apt_size_range = linspace(min(apt_sizes), max(apt_sizes), 10);
apt_rooms_range = linspace(min(apt_rooms), max(apt_rooms), 10);

% Calculate predictions for each possible combination of rooms number and appartment size.
apt_prices = zeros(length(apt_size_range), length(apt_rooms_range));
for apt_size_index = 1:length(apt_size_range)
  for apt_room_index = 1:length(apt_rooms_range)
      x = [1, apt_size_range(apt_size_index), apt_rooms_range(apt_room_index)];
      apt_prices(apt_size_index, apt_room_index) = hypothesis(x, theta);
  end
end

% Plot the plane on top of training data to see how it feets them.
subplot(2, 2, 2);
hold on;
mesh(apt_size_range, apt_rooms_range, apt_prices);
legend('Training Examples', 'Hypothesis Plane')
hold off;
