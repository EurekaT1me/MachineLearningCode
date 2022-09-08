function [best_acc,best_y_pre,best_p,best_q,best_S,obj,acc,p_lib,q_lib,S_lib]=FCDMF(B,p,q,YL,options)
%% parameter settings
MAX_ITER=options.MAX_ITER;
threshold=options.threshold;
a1=options.a1; % the minmun number of samples within each cluster
a2=options.a2; % the minmun number of features within each cluster

%% initialization
obj=zeros(MAX_ITER,1);
acc=[];
p_lib=[];
q_lib=[];
S_lib={};
P=to_categorical(p);
Q=to_categorical(q);
%% main code
for iter=1:MAX_ITER
   % update S
   S=update_S(P,B,Q);
   S_lib=[S_lib;S];
   % update P
   SQT=S*Q';
   DBP=compute_distance(B,SQT);
   p=update_pq(DBP,p,P,a1);
   P=to_categorical(p);
   p_lib=[p_lib;p];
   % update Q
   STPT=S'*P'; % In the paper, authors use the column of matrix but our compute_distance use the row of matrix to compute. So we need transpose matrix to correctly compute.
   BT=B';
   DBQ=compute_distance(BT,STPT);
   q=update_pq(DBQ,q,Q,a2);
   Q=to_categorical(q);
   q_lib=[q_lib;q];
   % compute accuracy
   acc=[acc;compute_acc(p',YL)];
   % compute objective value
   obj(iter)=compute_obj(B,P,Q,S);
   if iter>=2 && (abs(obj(iter)-obj(iter-1))/obj(iter-1))<threshold
       break;
   end
end
% find best accuracy and corresponding parameter
[best_acc,pos]=max(acc);
best_y_pre=p_lib(pos,:);
best_p=p_lib(pos,:);
best_q=q_lib(pos,:);
best_S=S_lib{pos};
end

function [Y]=to_categorical(y)
% y: the label vector
% Y: the discrete indicate matrix, one-hot form of label vector
c=length(unique(y));
Y=zeros(length(y),c);
for i=1:length(y)
    Y(i,y(i))=1;
end
end

function [D]=compute_distance(A,B)
% in each iteration, the function compute the distace between the i-the row of A
% and the k-the row of B
num=size(A,1);
c=size(B,1);
D=zeros(num,c);
for i=1:num
   for k=1:c
       D(i,k)=F22norm(A(i,:)-B(k,:));
   end
end
end

function [acc]=compute_acc(y_pre,y_true)
result=ClusteringMeasure(y_true,y_pre);
acc=result(1);
end

function [y]=update_pq(D,y,Y,L)
% D: the distance matrix
% y: the label vector of n samples
% Y: the discrete indicate matrix Y, n(or d)*c, one-hot form
% L: the minimun number of samples within each cluster

nc=diag(Y'*Y); % the number of samples within each cluster
while 1
   converge=1;
   for i=1:size(Y,1)
      c_old=y(i);
      if nc(c_old)<=L % if in one cluster, the number of samples is less than L, stop the following operation
          continue;
      end
      [~,min_idx]=min(D(i,:),[],2); % obtain the index of minimun distance value
      c_new=min_idx;
      if c_old~=c_new % c_old~=c_new means the model needs to update cluster indicate matrix
         y(i)=c_new;
         nc(c_old)=nc(c_old)-1;
         nc(c_new)=nc(c_new)+1;
         converge=0;
      end
   end
   if converge==1
      break; 
   end
end
end

function [val]=compute_obj(B,P,Q,S)
    val=F22norm(B-P*S*Q');
end

function [S]=update_S(P,B,Q)
% P: n*c, sample clustering indicate matrix
% Q: d*c, feature clustering indicate matrix
% B: n*d, data matrix
% S: S=(F'DF)^{-1}
PTBQ=P'*B*Q;
PTP=P'*P;
QTQ=Q'*Q;
temp_cc=diag(PTP)*diag(QTQ)';
S=PTBQ./temp_cc;

end