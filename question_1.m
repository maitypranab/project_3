
% Number of data points
N = 200;

% Generate random values of x uniformly distributed between -5 and 5
x = unifrnd(-5, 5,1,N);

% Parameters for the true model t = w0 + w1*x + w2*x^2 + n
w0 = 1;
w1 = -2;
w2 = 0.5;

% Generate random noise from N(0, 1)
n = normrnd(0,sqrt(1),[1,N]);

% Generate the true values of y using the quadratic model
y_true = w0 + w1 * x + w2 * x.^2 + n;

% Fit a linear model (degree 1) using least squares
linear_coefficients = polyfit(x, y_true, 1);

% Fit a quadratic model (degree 2) using least squares
quadratic_coefficients = polyfit(x, y_true, 2);

% Generate points for the fitted models
x_fit = linspace(-5, 5, 100);
y_linear_fit = polyval(linear_coefficients, x_fit);
y_quadratic_fit = polyval(quadratic_coefficients, x_fit);

% Plot the true data and the fitted models
scatter(x, y_true,20, 'DisplayName', 'True Data');
hold on;
plot(x_fit, y_linear_fit, 'r', 'DisplayName', 'Linear Fit');
plot(x_fit, y_quadratic_fit, 'k', 'DisplayName', 'Quadratic Fit');
hold off;

title('Linear and Quadratic Fits to Data');
xlabel('x');
ylabel('y');
legend('show');
