% Number of data points
N = 100;

% Generate N values of x uniformly distributed between -5 and 5
x = linspace(-5, 5, N)';

% Generate random noise with N(0, 300) distribution
n = normrnd(0, sqrt(300), size(x));

% Generate data based on the given equation: t = 5*x^3 - x^2 + x + n
t = 5 * x.^3 - x.^2 + x + n;

% Fit linear model
linear_coefficients = polyfit(x, t, 1);
linear_model = polyval(linear_coefficients, x);

% Fit cubic model
cubic_coefficients = polyfit(x, t, 3);
cubic_model = polyval(cubic_coefficients, x);

% Fit sixth-order model
sixth_order_coefficients = polyfit(x, t, 6);
sixth_order_model = polyval(sixth_order_coefficients, x);

% Calculate predictive error bars for all models
linear_error = sqrt(sum((t - linear_model).^2) / (N - 2));
cubic_error = sqrt(sum((t - cubic_model).^2) / (N - 4));
sixth_order_error = sqrt(sum((t - sixth_order_model).^2) / (N - 7));

% Plot the original data and the fitted models
figure;

% Plot for Linear Model Fitting
subplot(3, 1, 1);
plot(x, t, 'o'); % Plot original data points
hold on;
plot(x, linear_model, 'r-'); % Plot linear model
hold on;
errorbar(x, linear_model, linear_error, 'g.', 'LineStyle', 'none'); % Plot error bars
hold off;
title('Linear Model Fitting');
xlabel('x');
ylabel('t');
legend('Data', 'Linear model', 'Location', 'best');

% Plot for Cubic Model Fitting
subplot(3, 1, 2);
plot(x, t, 'o'); % Plot original data points
hold on;
plot(x, cubic_model, 'k-'); % Plot cubic model
errorbar(x, cubic_model, cubic_error, 'r.', 'LineStyle', 'none'); % Plot error bars
hold off;
title('Cubic Model Fitting');
xlabel('x');
ylabel('t');
legend('Data', 'Cubic model', 'Location', 'best');

% Plot for Sixth-Order Model Fitting
subplot(3, 1, 3);
plot(x, t, 'o'); % Plot original data points
hold on;
plot(x, sixth_order_model, 'm-'); % Plot sixth-order model
errorbar(x, sixth_order_model, sixth_order_error, 'm.', 'LineStyle', 'none'); % Plot error bars
hold off;
title('Sixth-Order Model Fitting');
xlabel('x');
ylabel('t');
legend('Data', 'Sixth order model', 'Location', 'best');
