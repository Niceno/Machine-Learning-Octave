clear;

%----------------------------------------
%
% Define a non-linear scattered data set
% x - features
% y - results
%
%----------------------------------------
m = 64;
r = rand(m,1);  % make it a vector (like a column)

%-----------------
% Define features
%-----------------

% Define data range
x_min = -3;
x_max = +3;
% Assign values in given data range ...
x = [x_min : (x_max-x_min)/(m-1) : x_max]';  % also vector / column

%-----------------------
% Create data (results)
%-----------------------
y = 2.0 + x + 0.5 * x.^2 ...  % this defines the shape of data
  + 4.0 * (r-0.5);            % this adds some random noise on top

order = input ("Enter polynomial order: ")

%-------------------------------------------
% Append the column with higher order terms
%-------------------------------------------
xl = x;  % original linear vector
for p = 2:order
  x = [x xl.^p];
end

% Plot the training data set
scatter(x(:,1), y);
title('Training Set');
xlabel('Feature');
ylabel('Results');
hold;

%--------------------
%
% Running regression
%
%--------------------
fprintf('Running regression...\n');

% Setup regularization parameter.
lambda =   0;
alpha  =   0.001;
num_i  = 10000;
[theta mu sigma x_normalized j_history] = ...
  regression_train(x,                     ...
                   y,                     ...
                   alpha,                 ...
                   lambda,                ...
                   num_i);

fprintf('- Initial cost:   %f\n', j_history(1));
fprintf('- Optimized cost: %f\n', j_history(end));

fprintf('- Theta (with normalization):\n');
fprintf('-- %f\n', theta);
fprintf('\n');

%----------------------------
%
% Check the solution you got
%
%----------------------------
y_h = hypothesis([ones(m,1) x_normalized], theta);
plot(x(:,1), y_h);
