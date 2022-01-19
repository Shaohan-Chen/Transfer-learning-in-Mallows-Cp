function [alpha_0,v_1,v_2,v_3]=the_original_TLCp_multisource_linearequations(X1,Y1,X2,Y2,X3,Y3,lambda_1,lambda_2,lambda_3,lambda_4,current_feature_number)

k=current_feature_number;

A11=2*lambda_1*X1'*X1;
A12=2*lambda_1*X1'*X1+lambda_4;
A13=zeros(k,k);
A14=zeros(k,k);

A21=2*lambda_2*X2'*X2;
A22=zeros(k,k);
A23=2*lambda_2*X2'*X2+lambda_4;
A24=zeros(k,k);

A31=2*lambda_3*X3'*X3;
A32=zeros(k,k);
A33=zeros(k,k);
A34=2*lambda_3*X3'*X3+lambda_4;

A41=zeros(k,k);
A42=eye(k,k);
A43=eye(k,k);
A44=eye(k,k);

A=[A11 A12 A13 A14;A21 A22 A23 A24;A31 A32 A33 A34;A41 A42 A43 A44];

b1=2*lambda_1*X1'*Y1;
b2=2*lambda_2*X2'*Y2;
b3=2*lambda_3*X3'*Y3;
b4=zeros(k,1);

b=[b1;b2;b3;b4];

solution=A\b;

alpha_0=solution(1:k);
v_1=solution(k+1:2*k);
v_2=solution(2*k+1:3*k);
v_3=solution(3*k+1:4*k);

