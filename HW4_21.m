clear;
Xtest=importdata('ratings_test.txt');
Xtrain=importdata('ratings.txt');
Xname=importdata('movies.txt');
Xpredict=zeros(5000,3);
RMSEplot=zeros(100,1);
LJLplot=zeros(100,1);

lamda=10;
sigma=0.5;
d=20;

N1=943;
N2=1682;
M=zeros(N1,N2);
U=zeros(N1,d);
V=zeros(d,N2);
I=eye(d);

for i=1:95000
    M(Xtrain(i,1),Xtrain(i,2))=Xtrain(i,3);
end

iteration=100;

%initializtion of Vj
V=normrnd(3,1,d,N2);%产生均值为a、方差为b大小为cXd的随机矩阵

for iter=1:iteration
    for i=1:N1
        sum1=zeros(d,d);
        sum2=zeros(d,1);
        for j=1:N2
            if (M(i,j)~=0)
                sum1=sum1+V(:,j)*transpose(V(:,j));
                sum2=sum2+M(i,j)*V(:,j);
            end
        end
        U(i,:)=pinv(lamda*sigma*sigma*I+sum1)*sum2;
               
    end
    for j=1:N2
        sum1=zeros(d,d);
        sum2=zeros(d,1);
        for i=1:N1
            if (M(i,j)~=0)
                sum1=sum1+transpose(U(i,:))*U(i,:);
                sum2=sum2+transpose(M(i,j)*U(i,:));
            end
        end
        V(:,j)=transpose(pinv(lamda*sigma*sigma*I+sum1)*sum2);
        
    end
    
    tmpSum=0;
    for i=1:5000
        Xpredict(i,3)=round(U(Xtrain(i,1),:)*V(:,Xtrain(i,2)));
        tmpSum=tmpSum+(Xpredict(i,3)-Xtest(i,3))*(Xpredict(i,3)-Xtest(i,3));
    end

    RMSE=sqrt(tmpSum/5000);
    RMSEplot(iter,1)=RMSE;
    
    LJLsum1=tmpSum/(2*sigma*sigma);
    
    LJLsum2=0;
    for i=1:N1
        LJLsum2=LJLsum2+norm(U(i,:))^2;
    end
    LJLsum2=LJLsum2*lamda/2;
    
    LJLsum3=0;
    for j=1:N2
        LJLsum3=LJLsum3+norm(V(:,j))^2;
    end
    LJLsum3=LJLsum3*lamda/2;
    LJLsum=LJLsum1+LJLsum2+LJLsum3;
    LJLplot(iter,1)=-LJLsum;
    
end

% for i=1:5000
%     Xpredict(Xtrain(i,1),Xtrain(i,2))=round(U(Xtrain(i,1),:)*V(:,Xtrain(i,2)));
% end

% for i=1:5000
%     sum=sum+(Xpredict(i,3)-Xtest(i,3))*(Xpredict(i,3)-Xtest(i,3));
% end

figure(1)
plot(RMSEplot);

figure(2)
plot(LJLplot);




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

randomCentroid=round(k*rand(1,5));
selectedCentroids=zeros(5,20);
multiplication=zeros(N2,5);
multiplicationSort=zeros(N2,5);
movieIndex=zeros(10,5);

for i=1:5
    selectedCentroids(i,:)=mu(randomCentroid(1,i),:);
end

for i=1:5
    for j=1:N2
        multiplication(i,j)=U(i,:)*V(:,j);
    end
end

for j=1:5
    multiplicationSort(:,j)=sort(multiplication(:,j));
    for i=1:10
        [x,y]=find(multiplication(:,j)==multiplicationSort(N2-i+1,j));
        movieIndex(i,j)=x;
    end
end

disp('5 centroid randomly chosen and their closest 10 movies are listed below : ');

for i=1:5
    disp(i);
    disp('the centroid coordinate is showed below: ')
    disp(mu(randomCentroid(1,i),1));
    disp(' 10 movies are : ');
    for j=1:10
        disp(Xname(movieIndex(j,i),1));
    end
end





