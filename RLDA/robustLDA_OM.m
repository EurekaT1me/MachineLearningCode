function [W, obj,DC] = robustLDA_OM(X,y,m)  
c = length(unique(y));
n = length(y);
for i=1:c
    Xc{i} = X(:,y==i);
    nc(i) = size(Xc{i},2);
    dc{i} = ones(nc(i),1);
end;
H = eye(n) - 1/n*ones(n);
St = X*H*X';
obj=[];
Sw = cell(0);
for iter = 1:15
    DC = [];
    M = 0;
    for i=1:c
        Xi = Xc{i};
        ni = nc(i);
        D = diag(1./dc{i});
        d = diag(D);
        mi = Xi*d/sum(d);
        Xmi = Xi-mi*ones(1,ni);
        Sw = [Sw,Xmi*D*Xmi'];
        M = M + Xmi*D*Xmi';
        Xm{i} = Xmi;
    end;
    St = max(St,St');
    M = max(M,M');
    p=pinv(St)*M;
    W = eig1(p,m,0,0);
    W = W*diag(1./sqrt(diag(W'*St*W)));
    for i=1:c
        Xmi = Xm{i};
        WX = W'*Xmi;
        dc{i} = sqrt(sum(WX.*WX,1)+eps);
        ob(i) = sum(dc{i});
        DC = [DC;dc{i}];
    end;

    obj(iter,1) = sum(ob);      
end