N2=1682;
randomMovie=round(N2*rand(1,3));

distanceMatrix=zeros(N2,3);
distanceMatrixSort=zeros(N2,3);
resultIndex=zeros(5,3);

for j=1:N2
    for i=1:3
        sum=0;
        for k=1:N2
            sum=sum+norm(V(:,j)-V(:,k))^2;
        end
        distanceMatrix(j,i)=sum;
    end
end

for j=1:3
    distanceMatrixSort(:,j)=sort(distanceMatrix(:,j));
    for i=1:5
        [x,y]=find(distanceMatrix(:,j)==distanceMatrixSort(i+1,j));
        resultIndex(i,j)=x;
    end
end

disp('3 movies randomly chosen and their nearest 5 neighbors are listed below : ')

for i=1:3
    disp(i);
    disp(Xname(randomMovie(1,i),1));
    disp(' : ');
    for j=1:5
        disp(Xname(resultIndex(j,i),1));
    end
end

