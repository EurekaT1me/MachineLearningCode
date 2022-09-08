function [Y] = initialY(X, c_true, rep, way)
% X: dara matrix, n*d
% c_true: the number of class
% rep: repeat times
% way: the way of initialization: 'random','k-means','k-means++'
if nargin < 4
    way = 'random';
end
[num, dim] = size(X);
Y = zeros(rep, num);
if strcmp(way, 'random')
    for i = 1:rep
        y1 = [1:c_true];
        y2 = randi(c_true, 1, num-c_true);
        y3 = [y1, y2];
        randidx = randperm(length(y3));
        y3_new = y3(randidx);
        Y(i,:)=y3_new;
    end
elseif strcmp(way, 'k-means')
    Y=my_kmeans(X,c_true,rep,'sample');
elseif strcmp(way, 'k-means++')
    Y=my_kmeans(X,c_true,rep,'plus');
else
    error('no such options in initialY');
end
end

function [Y]=my_kmeans(X,c,rep,init)
% X: 2D-array data matrix with size of N * dim
% YL: 1D-array label matrix
% c: the number of cluster
% rep: the number repeat runs
% init: the method of initialization
if nargin<5
   init='sample'; 
end
num=size(X,1);
Y=zeros(rep,num);
for i=1:rep
   idx=kmeans(X,c,'display','final','start',init);
   Y(i,:)=idx;
end


end
