
% Creating .mat file
xn = [1896,1900,1904,1906,1908,1912,1920:4:1936,1948:4:2008]';
tn = [12.0,11.0,11.0,11.2,10.8,10.8,10.8,10.6,10.8,10.3,10.3,10.3,10.4,10.5,10.2,10.0,9.95,10.14,10.06,10.25,9.99,9.92,9.96,9.84,9.87,9.85,9.69]';
%save olympicmen.mat xn tn
save('olympicmen.mat','xn','tn')

% Load Olympic men's 100m data
load olympicmen.mat;

% Set the number of folds for cross-validation
K = 5;

% Define the size of the dataset
N = numel(tn);

% Permute the data order randomly for cross-validation
order = randperm(N);

% Replicate the matrix of size N/K up to K times
sizes = repmat(floor(N/K), 1, K);

% Add remaining data to the last set
sizes(end) = sizes(end) + mod(N, K);

% Cumulative addition of sizes of data sets
sizes = [0 cumsum(sizes)];

% Rescale years for normalization
xn = xn - xn(1)./5;

% Create linear and fourth-order polynomial data sets
linear_data = [ones(length(xn)) xn];
fourth_order_data = [linear_data xn.^2 xn.^3 xn.^4];

% Regularization parameter values (modify the range as needed)
lamda_range = logspace(-10, 10, 21);

% Call the cross-validation function for the linear model
[~, linear_losses] = crossval(linear_data, order, lamda_range, K, tn, sizes, N);

% Call the cross-validation function for the fourth-order model
[~, fourth_order_losses] = crossval(fourth_order_data, order, lamda_range, K, tn, sizes, N);

% Find the best lambda for linear model
[~, idx_linear] = min(mean(linear_losses, 2));
best_lambda_linear = lamda_range(idx_linear);

% Find the best lambda for fourth-order model
[~, idx_fourth_order] = min(mean(fourth_order_losses, 2));
best_lambda_fourth_order = lamda_range(idx_fourth_order);

fprintf("Best lambda for Linear Model: %e\n", best_lambda_linear);
fprintf("Best lambda for Fourth-Order Model: %e\n", best_lambda_fourth_order);

% Cross-validation function definition
function [pred, losses] = crossval(X, order, lam, K, times, batch_size, N)
    losses = zeros(length(lam), K);

    for r = 1:length(lam)
        for k = 1:K
            % Extract the train and test data
            tr_values_times = times(order);
            tr_years = X(order, :);
            val_years = X(order(batch_size(k)+1:batch_size(k+1)), :); % Set validation data years
            val_output = times(order(batch_size(k)+1:batch_size(k+1))); % Set validation times
            tr_years(batch_size(k)+1:batch_size(k+1), :) = []; % Remove the validation part from train years
            tr_values_times(batch_size(k)+1:batch_size(k+1)) = []; % Remove the validation part from train times

            % Fit the model using pseudo-inverse
            temp_mat = pinv(tr_years' * tr_years + lam(r) * eye(size(X,2)));
            w = temp_mat * tr_years' * tr_values_times;

            % Compute loss on test data
            pred = val_years * w;
            losses(r, k) = sum((pred - val_output).^2);
        end
    end
end
