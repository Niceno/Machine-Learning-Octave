clear;

%----------------------------------------
%
% Define a non-linear scattered data set
% x - features
% y - results
%
%----------------------------------------
m = 512;
r = rand(m,1);  % make random numbers a vector (like a column)

%-----------------
% Define features
%-----------------

% Define data ranges
x1_min = -3;  x2_min = -4;
x1_max = +3;  x2_max = +4;

% Assign values in given data range ...
x1 = [x1_min : (x1_max-x1_min)/(m-1) : x1_max]';  % also vector / column
x2 = [x2_min : (x2_max-x2_min)/(m-1) : x2_max]';  % also vector / column

% ... and shuffle them to make it more interesting
x1 = x1(randperm(numel(x1)));
x2 = x2(randperm(numel(x2)));

%-----------------------
% Create data (results)
%-----------------------
y = 2.0             ...
  + 1.0 * x1        ...
  + 0.5 * x1.^2     ...
  - 0.2 * x2.^2     ...
  + 4.0 * (r-0.5);            % this adds some random noise on top
scatter3(x1, x2, y);

order = input ("Enter polynomial order: ")

%-------------------------------------------
% Append the column with higher order terms
%-------------------------------------------
xl1 = x1;       % original linear vectors
xl2 = x2;       % original linear vectors
x = [xl1 xl2];  % two features side by side in first order
for p = 2 : order
  x = [x  xl1.^p  xl2.^p];
end

% Plot the training data set
scatter3(x(:,1), x(:,2), y);
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

%-----------------------------------------
% In simple and stupid way (always works)
%-----------------------------------------
y_h = hypothesis([ones(m,1) x_normalized], theta);
scatter3(x(:,1), x(:,2), y_h);

%----------------------------------
% Plot in a more sophisticated way
%----------------------------------
x1s = sort(x1);  x1s = x1s(1:ceil(m/21):end);
x2s = sort(x2);  x2s = x2s(1:ceil(m/21):end);
[x1p, x2p] = meshgrid(x1s, x2s);
M = size(x1s,1);
x1s = reshape(x1p, M*M, 1);
x2s = reshape(x2p, M*M, 1);
xl1 = x1s;       % original linear vectors
xl2 = x2s;       % original linear vectors
xs = [xl1 xl2];  % two features side by side in first order
for p = 2 : order
  xs = [xs  xl1.^p  xl2.^p];
end
[x_normalized mu sigma] = feature_normalize(xs);
y_h = hypothesis([ones(M*M,1) x_normalized], theta);
yp = reshape(y_h, M, M);
surf(x1p,x2p,yp);

