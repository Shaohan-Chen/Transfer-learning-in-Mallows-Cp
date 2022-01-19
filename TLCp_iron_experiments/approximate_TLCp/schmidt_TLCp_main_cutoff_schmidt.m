%function [explained_variance_cutoff,explained_variance]=schmidt_TLCp_main_cutoff_schmidt(num_train_1)
tic
clear
num_test_1=100;
num_train_1=290;
%num_train_2=800;

data_1=xlsread('furnace_A.xlsx');

size_data=size(data_1);   % cancel data without complete information
j=1;
for i=1:size_data(1)
    if  min(abs(data_1(i,:))) ~= 0
        data_1_selected(j,:)=data_1(i,:);
       j=j+1;
   end
end

%rearrange the order of the data
r=randperm(size(data_1_selected,1));
data_11=data_1_selected(r,:);

%training data
X_1=data_11(1:num_train_1,2:end);
Y_1=data_11(1:num_train_1,1);

% training data standardized
X_1_normalization=(X_1-mean(X_1))./(std(X_1));
Y_1_normalization=(Y_1-mean(Y_1))/(std(Y_1));
X_train1=[X_1_normalization];
Y_train_1=Y_1_normalization;

%test data
X_2=data_11(num_train_1+1:num_train_1+num_test_1,2:end);
Y_2=data_11(num_train_1+1:num_train_1+num_test_1,1);

%test data standardized
X_2_normalization=(X_2-mean(X_1))./(std(X_1));
X_test_1=[X_2_normalization];
Y_test_1=Y_2;

% source domain data
data_2=xlsread('furnace_B.xlsx');
num_train_2=size(data_2,1);
X_train_2=data_2(1:num_train_2,2:end);
Y_train_2=data_2(1:num_train_2,1);

% data standardized
X_train_2_normalization=(X_train_2-mean(X_train_2))./std(X_train_2);
Y_train_2=(Y_train_2-mean(Y_train_2))/(std(Y_train_2)); 
X_train2=[X_train_2_normalization];

% delete some correlated features

data1(:,1)=X_train1(:,1); 
data2(:,1)=X_train2(:,1);
data11(:,1)=X_test_1(:,1);

h=2;
for i=2:size(X_train1,2)
     A=[X_train1(:,i),data1];
     D=[X_train2(:,i),data2];
     if rank(D)==h && rank(A)==h
        data1(:,h)=X_train1(:,i); 
        data2(:,h)=X_train2(:,i);
        data11(:,h)=X_test_1(:,i);
        h=h+1;
     end
end

% data rearrange
beta1=(data1'*data1)^(-1)*data1'*Y_train_1;
beta2=(data2'*data2)^(-1)*data2'*Y_train_2;
     
delta_test=beta1-beta2;
     
f=(abs(beta1)+abs(beta2)).*abs((1./delta_test));
     
[B,I]=sort(f,'descend');
     
data1=data1(:,I);
data2=data2(:,I);
     
% data orthogonalization

[Q_hat_1,data_schmidt1]=schmidt_orthogonalization_new(data1,num_train_1);
[Q_hat_2,data_schmidt2]=schmidt_orthogonalization_new(data2,num_train_2);
   
%% feature selection

beta_ols_1=(data_schmidt1'*data_schmidt1)^(-1)*data_schmidt1'*Y_train_1;
estimated_sigma_squared_1=(Y_train_1-data_schmidt1*beta_ols_1)'*(Y_train_1-data_schmidt1*beta_ols_1)/(num_train_1-size(data_schmidt1,2));

beta_ols_2=(data_schmidt2'*data_schmidt2)^(-1)*data_schmidt2'*Y_train_2;
estimated_sigma_squared_2=(Y_train_2-data_schmidt2*beta_ols_2)'*(Y_train_2-data_schmidt2*beta_ols_2)/(num_train_2-size(data_schmidt2,2));

delta=beta_ols_1-beta_ols_2;

for i=1:size(delta)
    v(i)=(4*estimated_sigma_squared_1*estimated_sigma_squared_2)/delta(i)^2;
end

% model parameters

lambda_1=estimated_sigma_squared_2;
lambda_2=estimated_sigma_squared_1;
lambda_3=v;
        for i=1:size(delta)
            D_1(i)=(lambda_2*v(i))/(4*lambda_1*lambda_2*num_train_1+lambda_2*v(i)+(num_train_1/num_train_2)*lambda_1*v(i));
            D_2(i)=(lambda_1*v(i))/(4*lambda_1*lambda_2*num_train_2+lambda_1*v(i)+(num_train_2/num_train_1)*lambda_2*v(i));
            D_3(i)=(2*lambda_1*lambda_2)/(4*lambda_1*lambda_2+(1/num_train_1)*lambda_2*v(i)+(1/num_train_2)*lambda_1*v(i));
            tilde_D(i)=lambda_1*num_train_1*(D_1(i))^2+lambda_2*num_train_2*(D_2(i))^2+v(i)*(D_3(i))^2;
            Q(i)=-2*tilde_D(i);
            N(i)=-tilde_D(i)+lambda_1*num_train_1;
            M(i)=-tilde_D(i)+lambda_2*num_train_2;
            G(i)=sqrt((num_train_2*num_train_1)/((num_train_1*M(i)*estimated_sigma_squared_2)+(num_train_2*N(i)*estimated_sigma_squared_1)));
            estimated_lambda_4(i)=(2*estimated_sigma_squared_1*(2-Q(i)/(sqrt(M(i)*N(i)))))/(4*estimated_sigma_squared_1*G(i)^2);
        end
lambda_4=min(estimated_lambda_4);

% Calculate the regression coefficients of target task;
[trained_beta_target]=Cp_transfer_solution(data_schmidt1,Y_train_1,data_schmidt2,Y_train_2,lambda_1,lambda_2,lambda_3,lambda_4,num_train_1,num_train_2,size(data_schmidt1,2));
trained_beta_hat_T1=Q_hat_1*trained_beta_target';

% calculate the regression coefficints of source task;
[trained_beta_source,tilde_v2]=Cp_transfer_source_solution(data_schmidt1,Y_train_1,data_schmidt2,Y_train_2,lambda_1,lambda_2,lambda_3,lambda_4,num_train_1,num_train_2,size(data_schmidt1,2));
trained_beta_hat_T2=Q_hat_2*trained_beta_source';


% parameter estimations for TLCp criterion
feature_number=size(data1,2);

beta1=(data1'*data1)^(-1)*data1'*Y_train_1;
beta2=(data2'*data2)^(-1)*data2'*Y_train_2;
     
tilde_lambda_2=(Y_train_1-data1*beta1)'*(Y_train_1-data1*beta1)/(num_train_1-feature_number);
tilde_lambda_1=(Y_train_2-data2*beta2)'*(Y_train_2-data2*beta2)/(num_train_2-feature_number);
estimated_sigma_squared_1=tilde_lambda_2;
estimated_sigma_squared_2=tilde_lambda_1;
     
delta_original_TLCp=beta1-beta2;
     
for i=1:feature_number
    v(i)=(4*estimated_sigma_squared_1*estimated_sigma_squared_2)/(delta_original_TLCp(i)^2);
end

tilde_lambda_3=v;

for i=1:feature_number
            D_1(i)=(tilde_lambda_2*v(i))/(4*tilde_lambda_1*tilde_lambda_2*num_train_1+tilde_lambda_2*v(i)+(num_train_1/num_train_2)*tilde_lambda_1*v(i));
            D_2(i)=(tilde_lambda_1*v(i))/(4*tilde_lambda_1*tilde_lambda_2*num_train_2+tilde_lambda_1*v(i)+(num_train_2/num_train_1)*tilde_lambda_2*v(i));
            D_3(i)=(2*tilde_lambda_1*tilde_lambda_2)/(4*tilde_lambda_1*tilde_lambda_2+(1/num_train_1)*tilde_lambda_2*v(i)+(1/num_train_2)*tilde_lambda_1*v(i));
            tilde_D(i)=tilde_lambda_1*num_train_1*(D_1(i))^2+tilde_lambda_2*num_train_2*(D_2(i))^2+v(i)*(D_3(i))^2;
            Q(i)=-2*tilde_D(i);
            N(i)=-tilde_D(i)+tilde_lambda_1*num_train_1;
            M(i)=-tilde_D(i)+tilde_lambda_2*num_train_2;
            G(i)=sqrt((num_train_2*num_train_1)/((num_train_1*M(i)*estimated_sigma_squared_2)+(num_train_2*N(i)*estimated_sigma_squared_1)));
            estimated_lambda_4(i)=(2*estimated_sigma_squared_1*(2-Q(i)/(sqrt(M(i)*N(i)))))/(4*estimated_sigma_squared_1*G(i)^2);
end
tilde_lambda_4=min(estimated_lambda_4);

% select threshold of u;
for h=1:feature_number
    u(h)=abs(trained_beta_hat_T1(h))/sqrt(estimated_sigma_squared_1*sum(Q_hat_1(h,:).^2)/num_train_1);
end
%The bigger the value of u1, the less features to be selected;
u_1=sort(u);
u1=[u_1,u_1(end)+1];

error_train_best=tilde_lambda_1*Y_train_1'*Y_train_1+tilde_lambda_2*Y_train_2'*Y_train_2;
optimal_trained_beta_hat_1=zeros(feature_number,1);
for m=1:size(u1,2)
        for k=1:feature_number
            tau(k)=sqrt(estimated_sigma_squared_1*sum(Q_hat_1(k,:).^2)/num_train_1)*u1(m); % cut-off value for each feature;
            if abs(trained_beta_hat_T1(k))>=tau(k)
               trained_beta_hat_1(k)=trained_beta_hat_T1(k);
               trained_beta_hat_2(k)=trained_beta_hat_T2(k);
            else
                trained_beta_hat_1(k)=0;
                trained_beta_hat_2(k)=0;
            end
        end
        if size(trained_beta_hat_1,1)==1
           trained_beta_hat_1=trained_beta_hat_1';
        end
        if size(trained_beta_hat_2,1)==1
           trained_beta_hat_2=trained_beta_hat_2';
        end
        count=sum(trained_beta_hat_1~=0);
    %objective_value=tilde_lambda_1*(Y_train_1-data1*trained_beta_hat_1)'*(Y_train_1-data1*trained_beta_hat_1)+tilde_lambda_2*(Y_train_2-data2*trained_beta_hat_2)'*(Y_train_2-data2*trained_beta_hat_2)+0.5*(norm(sqrt(tilde_lambda_3)*v1,2)^2+norm(sqrt(tilde_lambda_3)*v2,2)^2)+tilde_lambda_4*count;
    objective_value=tilde_lambda_1*(Y_train_1-data1*trained_beta_hat_1)'*(Y_train_1-data1*trained_beta_hat_1)+tilde_lambda_2*(Y_train_2-data2*trained_beta_hat_2)'*(Y_train_2-data2*trained_beta_hat_2)+tilde_lambda_4*count;
    %objective_value=(Y_train_1-data1*trained_beta_hat_1)'*(Y_train_1-data1*trained_beta_hat_1)+2*estimated_sigma_squared_1*count; 
    error_train=objective_value;
       if error_train<error_train_best
          error_train_best=error_train;
          u_best=u1(m);
          optimal_trained_beta_hat_1=trained_beta_hat_1;
       end
end
 
trained_beta_hat_1=optimal_trained_beta_hat_1;

% intercept term

 beta0_cutoff=mean(Y_train_1-data1*trained_beta_hat_1);
 beta0=mean(Y_train_1-data1*trained_beta_hat_T1);

 % estimated beta rearrange
 for a=1:feature_number
     trained_beta_hat11(a,1)=trained_beta_hat_1(find(I==a));
     trained_beta_hat_T11(a,1)=trained_beta_hat_T1(find(I==a));
 end
     
trained_beta_hat_1=trained_beta_hat11;
trained_beta_hat_T1=trained_beta_hat_T11;

%% error calculation
coef_cutoff=trained_beta_hat_1;
coef0_cutoff=beta0_cutoff;
coef=trained_beta_hat_T1;
coef0=beta0;
num_data=size(data_11,1);%total number of data
Y1=data_11(:,1);
total_variance=sum((Y1-mean(Y1)).^2)/num_data; % for total data
%non-cutoff version of approximate Cp
error=(Y_test_1-((data11*coef+coef0).*std(Y_1)+mean(Y_1)))'*(Y_test_1-((data11*coef+coef0).*std(Y_1)+mean(Y_1)));% only for test data
error_test=error/num_test_1;
explained_variance=1-error_test/total_variance;
%cutoff version of approximate Cp
error_cutoff=(Y_test_1-((data11*coef_cutoff+coef0_cutoff).*std(Y_1)+mean(Y_1)))'*(Y_test_1-((data11*coef_cutoff+coef0_cutoff).*std(Y_1)+mean(Y_1)));% only for test data
error_test_cutoff=error_cutoff/num_test_1;
explained_variance_cutoff=1-error_test_cutoff/total_variance;
toc
