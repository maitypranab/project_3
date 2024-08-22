% Setting up the data
N = 6;
x = linspace(0, 1, N)';
n = normrnd(0, sqrt(3), size(x));
t = 2.^x - 3 + n;

% Generating polynomial features for the model
testx = [0:0.01:1]';
X = [];
testX = [];
for k = 0:5
    X = [X x.^k];
    testX = [testX testx.^k];
end

% Regularization parameter values
lam = [0 1e-6 1e-2 1e-1];

% Plotting the original data points
plot(x, t, 'b.', 'markersize', 20);
hold on;

% Looping through different regularization parameters
for l = 1:length(lam)
    lambda = lam(l);
    
    % Computing the weights using Regularized least squares
    N = size(x, 1);
    w = (X' * X + N * lambda * eye(size(X, 2))) \ X' * t;
    
    % Plotting the model fit for each lambda
    plot(testx, testX * w, 'linewidth', 1.5)
    
    % Setting plot properties
    xlim([-0.1 1.1])
    title('5th order model fit for different \lambda ')
    xlabel('x', 'FontWeight', 'bold', 'FontSize', 10)
    ylabel('t', 'FontWeight', 'bold', 'FontSize', 10)
    
    % Adding legend for each lambda value
    legend('Data set', '\lambda=0', '\lambda=10^{-6}', '\lambda=0.01', '\lambda=0.1', 'FontWeight', 'bold', 'Location', 'best')
    
    % Ensuring legend and plot details are retained for each iteration
    hold on
end

% Clearing the 'hold' state to prevent further plots from being added
hold off
