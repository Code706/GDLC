function [LABEL, acc, NMI, obj_NMF] = GDLC(X, nClass, gnd)

% Low-dimensional matrix dimension
C=2;

% Number of  clusters
K=nClass;

rand('twister',5489);
% Obtain the number of rows and columns for the objective matrix


[mFea,nSmp]=size(X);

%initialization
% w1=rand(mFea,C);
% q1=rand(nSmp,C);
% m1=rand(mFea,1);
% n1=rand(nSmp,1);

w=logsig(rand(mFea,C));
q=logsig(rand(nSmp,C));
m=logsig(rand(mFea,1));
n=logsig(rand(nSmp,1));


Round = 0;

% Parameter Setting
eta=0.0035;
alpha1=0.1;
beta1=alpha1;
beta2=0.1;
alpha2=beta2;


%initialization
while Round < 10
Round = Round+1;

% Nonlinear Constrained NMF(NNMF)
for i=1:mFea      % Traversing the objective matrix
    for j=1:nSmp  % Traversing the objective matrix
        sum1=0;
        
        for kk=1:C
          sum1=sum1+ 1/(1+exp(-w(i,kk)))*1/(1+exp(-q(j,kk)));
        end        
        temp= X(i,j)-sum1-1/(1+exp(-m(i)))-1/(1+exp(-n(j)));
        
        % Update bias
        C1=1/(1+exp(-m(i)));
        m(i)=m(i)-eta*(temp*(-1)*C1*(1-C1)+alpha1*C1*(1-C1)*C1);
        D=1/(1+exp(-n(j)));
        n(j)=n(j)-eta*(temp*(-1)*D*(1-D)+alpha2*D*(1-D)*D);
        
        % Update the elements in the low-dimensional matrix
        for kk=1:C  
            A=1/(1+exp(-w(i,kk)));
            w(i,kk)=w(i,kk)-eta*(temp*(-1)*1/(1+exp(-q(j,kk)))*A*(1-A)+beta1*A*(1-A)*A);
        end
        
        for kk=1:C
            B=1/(1+exp(-q(j,kk)));
            q(j,kk)=q(j,kk)-eta*(temp*(-1)*1/(1+exp(-w(i,kk)))*B*(1-B)+beta2*B*(1-B)*B);
        end
    end 
end

% Generalized Deep Learning clustering (GDLC) updates all elements in w,q,m,n
% Element updates in NNMF are transformed into generalized weights and generalized biases.
w=logsig(w);% Eq.(23) 
q=logsig(q);% Eq.(24) 
m=logsig(m);% Eq.(25) 
n=logsig(n);% Eq.(26) 

% Calculate the value of the objective function
Q=q; W=w; m1=m; n1=n; 
Ux=[W,m1];Vx=[Q,n1];
dX = Ux*Vx'-X;
obj_NMF1(Round) = sum(sum(dX.^2));
obj_NMF2(Round) = beta1*sum(sum(W.^2))+beta2*sum(sum(Q.^2));
bojbias3(Round)=   alpha1*sum(m.^2)+alpha2*sum(n.^2);
obj_NMF4(Round)=obj_NMF1(Round)+obj_NMF2(Round)+bojbias3(Round);
obj_NMF(Round)=sqrt(obj_NMF4(Round));


% Obtain clustering accuracy
rand('twister',5489);
label = litekmeans(Q,K,'Replicates',20);
idx22 = bestMap(gnd,label);
LABEL{Round} = idx22;
acc(Round)= length(find(gnd == idx22))/length(gnd);%accuracy
NMI(Round) = MutualInfo(gnd,label);
disp(['Number of Rounds:' num2str(Round)])
disp(['Object Function Value: ',num2str(obj_NMF(Round))]);
disp(['Acc: ',num2str(acc(Round))]);
disp(['NMI: ',num2str(NMI(Round))]);
disp('-------------------------------------------------------');

end

end

%==========================================================================%



        