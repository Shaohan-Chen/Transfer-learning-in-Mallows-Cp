function theta=nng_new(Z,Y,lambda,options)

d=size(Z,2);
H =Z'*Z;
f=lambda*ones(1,d)-Y'*Z;
A=[];
b=[];
Aeq=[];
beq=[];
LB=zeros(d,1);
UB=1000*ones(d,1);%Inf*ones(d,1);
theta=quadprog(H,f,A,b,Aeq,beq,LB,UB,[],options);