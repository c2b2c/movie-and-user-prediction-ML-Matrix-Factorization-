clear;
%1 Generate 500 points

mu1 = [0 0];       % Mean
sigma1 = [ 1 .0;   % Covariance matrix
          .0  1];
m1 = 100;          % Number of data points

mu2 = [3 0];
sigma2 = [1 0;
          0 1];
m2 = 250;

mu3 = [0 3];
sigma3 = [1 0;
          0 1];
m3 = 150;

R1 = chol(sigma1);
X1 = randn(m1, 2) * R1;
X1 = X1 + repmat(mu1, size(X1, 1), 1);

R2 = chol(sigma2);
X2 = randn(m2, 2) * R2;
X2 = X2 + repmat(mu2, size(X2, 1), 1);

R3 = chol(sigma3);
X3 = randn(m3, 2) * R3;
X3 = X3 + repmat(mu3, size(X3, 1), 1);

X = [X1; X2; X3];

%2 Plot original data

figure(1);

hold off;
plot(X1(:, 1), X1(:, 2), 'bo');
hold on;
plot(X2(:, 1), X2(:, 2), 'ro');
hold on;
plot(X3(:, 1), X3(:, 2), 'go');

set(gcf,'color','white') 

gridSize = 100;
u = linspace(-6, 6, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

z1 = gaussianND(gridX, mu1, sigma1);
z2 = gaussianND(gridX, mu2, sigma2);
z3 = gaussianND(gridX, mu3, sigma3);

Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);
Z3 = reshape(z3, gridSize, gridSize);

[C, h] = contour(u, u, Z1);
[C, h] = contour(u, u, Z2);
[C, h] = contour(u, u, Z3);

axis([-6 6 -6 6])
title('Original Data and their PDFs');

%3 Initiation of the dataset

m = size(X, 1);

k = 3; 
n = 2;  % vector lengths.

% k random initial means
indeces = randperm(m);
mu = X(indeces(1:k), :);

sigma = [];

% take overal covariance of the dataset as initial variance for each cluster.
for (j = 1 : k)
    sigma{j} = cov(X);
end

% assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1 / k);

% 4: Expectation Maximization

W = zeros(m, k);

% iteration=20
objectPlot= zeros(20,2);

for iter = 1:20
    
    fprintf('Iteration %d\n ', iter);

    pdf = zeros(m, k);
    
    % For each cluster...
    for j = 1 : k
        
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(X, mu(j, :), sigma{j});
    end
    
    % multiply each pdf value by the prior probability for cluster.
    pdf_w = bsxfun(@times, pdf, phi);
    
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
    
    % do maximization   

    % For each of the clusters...
    for j = 1 : k
    
        % Calculate the prior probability for cluster 'j'.
        phi(j) = mean(W(:, j), 1);
        mu(j, :) = weightedAverage(W(:, j), X);
        sigma_k = zeros(n, n);
        Xm = bsxfun(@minus, X, mu(j, :));
        
        % Calculate the contribution of each training example to the covariance matrix.
        for i = 1 : m
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));
        end
        
        % Divide by the sum of weights.
        sigma{j} = sigma_k ./ sum(W(:, j));
    end
    
%     objectValue=0;
%     for i=1:500
%         tmpValue=0;
%         for j=1:k
%             tmpValue=tmpValue+W(i,j)*phi(j);
%         end
%         log(tmpValue)
%         objectValue=objectValue+log(tmpValue);
%     end
    
    objectValue=0;
    for i=1:500
        objectValue=objectValue+log(max(max(W(i,:))));
    end
    
    objectPlot(iter,1)= objectValue;
    objectPlot(iter,2)= iter;
%     figure(iter+2);
%     plot(objectPlot(:,2), objectPlot(:,1), 'bo');
    
    %fprintf(sigma{j});
      
end

figure(2);
color=['bo','ro','go','yo','po',];
for i=1:500
    [x y]=find(W(i,:)==max(max(W(i,:))));
    cluster=y;
    if (cluster==1)
        plot(X(i, 1), X(i, 2), 'bo');
    end
    if (cluster==2)
        plot(X(i, 1), X(i, 2), 'ro');
    end
    if (cluster==3)
        plot(X(i, 1), X(i, 2), 'go');
    end
    if (cluster==4)
        plot(X(i, 1), X(i, 2), 'ko');
    end
    if (cluster==5)
        plot(X(i, 1), X(i, 2), 'mo');
    end
    hold on;
end
axis([-6 6 -6 6])
title('Original Data and Estimated PDFs');
set(gcf,'color','white') % white background

figure(3);
plot(objectPlot(:,2), objectPlot(:,1), 'bo');
title('Objective function plot');
