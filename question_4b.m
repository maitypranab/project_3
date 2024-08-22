% Define mean and covariance matrix
mu = [2 1];                  % Mean vector
sigma = [1 0.8; 0.8 1];      % Covariance matrix

% Create a grid of points
[x, y] = meshgrid(-4:0.1:4, -4:0.1:4);
xy = [x(:) y(:)];

% Calculate the PDF values for each point in the grid
pdf_values = mvnpdf(xy, mu, sigma);

% Reshape the PDF values to match the grid size
pdf_matrix = reshape(pdf_values, size(x));

% Plot the PDF surface
figure;
subplot(1,2,1)
surf(x, y, pdf_matrix);
title('PDF of 2D Gaussian Random Vector');
xlabel('X-axis');
ylabel('Y-axis');
zlabel('PDF');
subplot(1,2,2)
contour(pdf_matrix);
title('Contour plot of 2D Gaussian Random Vector')
