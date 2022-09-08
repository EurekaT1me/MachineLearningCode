clear
clc
addpath('./func');
% load data
load('ORL.mat');
% call model
X=double(X);
Y=double(y_true);
options.MAX_ITER=100;
options.threshold=1e-8;
rep=10; % repeat times
way='k-means++'; % the method of initialization of P and Q

fprintf(strcat(repmat('=',1,20),'Initialization Begin',repmat('=',1,20),'\n'));
c_true=length(unique(Y));
[U,S,V]=svds(X,c_true);
P=initialY(U(:,1:c_true),c_true,rep,way);
Q=initialY(V(:,1:c_true),c_true,rep,way);
fprintf(strcat(repmat('=',1,20),'Initialization end',repmat('=',1,20),'\n'));
result=opt(P,Q,c_true,X,Y,options);

function [result]=opt(initP,initQ,c_true,X,Y_true,options)
[rep,num]=size(initP);
[~,dim]=size(initQ);
Y=zeros(rep,num);
options.a1=max(floor(num/10/c_true),1);
options.a2=max(floor(dim/10/c_true),1);
result.best_acc=zeros(rep,1);
result.best_y_pre={};
result.best_p={};
result.best_q={};
result.best_S={};
result.obj={};
result.acc={};
result.p_lib={};
result.q_lib={};
result.S_lib={};
for i =1:rep
    fprintf(strcat(repmat('=',1,20),'repeat=',num2str(i),repmat('=',1,20),'\n'));
    p=initP(i,:);
    q=initQ(i,:);
    [best_acc,best_y_pre,best_p,best_q,best_S,obj,acc,p_lib,q_lib,S_lib]=FCDMF(X,p,q,Y_true,options);
    result.best_acc(i)=best_acc;
    result.best_y_pre=[result.best_y_pre;best_y_pre];
    result.best_p=[result.best_p;best_p];
    result.best_q=[result.best_q;best_q];
    result.best_S=[result.best_S;best_S];
    result.obj=[result.obj;obj];
    result.acc=[result.acc;acc];
    result.p_lib=[result.p_lib;p_lib];
    result.q_lib=[result.q_lib;q_lib];
    result.S_lib=[result.S_lib;S_lib];
       
end
end






