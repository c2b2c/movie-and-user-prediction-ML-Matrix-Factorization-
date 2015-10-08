X = U;

%3 Initiation of the dataset

m = size(X, 1);

k = 30; 
n = 20;  % vector lengths.

% k random initial means
indeces = randperm(m);
mu = X(indeces(1:k), :);

sigma = [];

% take overal covariance of the dataset as initial variance for each cluster.
for j = 1 : k
    sigma{j} = cov(X);
end

% assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1 / k);

% 4: Expectation Maximization

W = zeros(m, k);

% iteration=20

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

randomCentroid=round(k*rand(1,5));
selectedCentroids=zeros(5,20);
multiplication=zeros(N2,5);
multiplicationSort=zeros(N2,5);
movieIndex=zeros(10,5);


%randomCentroid(:)

for i=1:5
    selectedCentroids(i,:)=mu(randomCentroid(1,i),:);
end

for i=1:5
    for j=1:N2
        multiplication(j,i)=U(i,:)*V(:,j);
    end
end


for j=1:5
    multiplicationSort(:,j)=sort(multiplication(:,j));
    for i=1:10
        [x,y]=find(multiplication(:,j)==multiplicationSort(N2-i+1,j));
        movieIndex(i,j)=x;
    end
end

disp('5 centroid randomly chosen £¬their indexes are: (among 30 cluster)');

disp(randomCentroid(:));

disp('Their closest 10 movies are listed below : ');

for i=1:5
    disp('Centroid ');
    disp(i);
%     disp('the centroid coordinate is showed below: ')
%     disp(mu(randomCentroid(1,i),:)*100000);
    disp(' 10 movies are : ');
    for j=1:10
        disp(Xname(movieIndex(j,i),1));
    end
end
