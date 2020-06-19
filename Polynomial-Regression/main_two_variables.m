clear;

%----------------------------------------
%
% Define a non-linear scattered data set
% x - features
% y - results
%
%----------------------------------------
m = 512;
n =   2;
r = rand(m, 1);  % make random numbers a vector (like a column)

%-----------------
% Define features
%-----------------

% Define data ranges
x_min = [-3, -4];
x_max = [+3, +4];

% Assign values in given data range ...
x = zeros(m, n);
for j = 1 : n
  x(:,j) = [x_min(j) : (x_max(j)-x_min(j))/(m-1) : x_max(j)]';
end

% ... and shuffle them to make it more interesting
for j = 1 : n
  x(:,j) = x( randperm( size(x,1) ), j);
end

%-----------------------
% Create data (results)
%-----------------------
y = 2.0                     ...
  + 1.0 * x(:,1)            ...
  + 0.5 * x(:,1).^2         ...
% - 2.0 * cos(x(:,2))       ...
  - 0.2 * x(:,2).^2         ...
  + 1.5 * x(:,1) .* x(:,2)  ...  % add a mixed term
  + 4.0 * (r-0.5);               % this adds some random noise on top
scatter3(x(:,1), x(:,2), y);

order = input ("Enter polynomial order: ")

%-------------------------------------------
% Append the column with higher order terms
%-------------------------------------------
xl = x;  % original linear vectors
for p = 2 : order
  x = [x  xl(:,1).^p  xl(:,2).^p];              % straight terms
  for q = 1 : p-1
    x = [x  (xl(:,1).^(p-q)) .* (xl(:,2).^q)];  % mixed terms
  end
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
xs = [sort(x(:,1)) sort(x(:,2))];       % sorted features
xt(:,1:2) = xs(1:ceil(m/41):end, 1:2);  % sorted and sifted features
[x1p, x2p] = meshgrid(xt(:,1), xt(:,2));
M = size(xt,1);
xs = [reshape(x1p, M*M, 1) reshape(x2p, M*M, 1)];
xl = xs;                                % store original linear vectors
for p = 2 : order
  xs = [xs  xl(:,1).^p  xl(:,2).^p];              % straight terms
  for q = 1 : p-1
    xs = [xs  (xl(:,1).^(p-q)) .* (xl(:,2).^q)];  % mixed terms
  end
end
[x_normalized mu sigma] = feature_normalize(xs);
y_h = hypothesis([ones(M*M,1) x_normalized], theta);
yp = reshape(y_h, M, M);
surf(x1p,x2p,yp);

