%function [explained_variance,explained_variance_cutoff]=Cp_transfer_multisource_new(num_train_1)
clear
tic
num_train_1=110;
num_test_1=30;

% data preparation
data=load('parkinsons_updrs copy.txt');
% using the data of the second patient as the target data;
j=1;
for i=1:size(data,1)

    if data(i,1)==1
       tilde_data(j,:)=data(i,:);
       j=j+1;
    end
end

%rearrange the order of the data
r=randperm(size(tilde_data,1));
data_1=tilde_data(r,:);

%training data

X_1=data_1(1:num_train_1,[4,7:end]);
Y_1=data_1(1:num_train_1,5);

% standardize training data
X_1_normalization=(X_1-mean(X_1))./std(X_1);
Y_1_normalization=(Y_1-mean(Y_1))/(std(Y_1));
X_train1=[X_1_normalization];
Y_train_1=Y_1_normalization;

%test data
X_2=data_1(num_train_1+1:num_train_1+num_test_1,[4,7:end]);
Y_2=data_1(num_train_1+1:num_train_1+num_test_1,5);

%test data standardized
X_2_normalization=(X_2-mean(X_1))./(std(X_1));
X_test_1=[X_2_normalization];
Y_test_1=Y_2;

% source domain data

j=1;
for i=1:size(data,1)

    if data(i,1)==24
       data_2(j,:)=data(i,:);
       j=j+1;
   end
end


num_train_2=size(data_2,1);
X_train_2=data_2(1:num_train_2,[4,7:end]);
Y_train_2=data_2(1:num_train_2,5);

% data standardized
X_train_2_normalization=(X_train_2-mean(X_train_2))./std(X_train_2);
Y_train_2=(Y_train_2-mean(Y_train_2))/(std(Y_train_2)); 
X_train2=[X_train_2_normalization];

% the second source task

j=1;
for i=1:size(data,1)

    if data(i,1)==33
       data_3(j,:)=data(i,:);
       j=j+1;
   end
end


num_train_3=size(data_3,1);
X_train_3=data_3(1:num_train_3,[4,7:end]);
Y_train_3=data_3(1:num_train_3,5);

% data standardized
X_train_3_normalization=(X_train_3-mean(X_train_3))./std(X_train_3);
Y_train_3=(Y_train_3-mean(Y_train_3))/(std(Y_train_3)); 
X_train3=[X_train_3_normalization];

% delete some correlated features

data1(:,1)=X_train1(:,1); 
data11(:,1)=X_test_1(:,1);
data2(:,1)=X_train2(:,1);
data3(:,1)=X_train3(:,1);

h=2;
for i=2:size(X_train1,2)
     A=[X_train1(:,i),data1];
     E=[X_train2(:,i),data2];
     F=[X_train3(:,i),data3];
     if rank(E)==h && rank(A)==h && rank(F)==h
        data1(:,h)=X_train1(:,i); 
        data11(:,h)=X_test_1(:,i); 
        data2(:,h)=X_train2(:,i); 
        data3(:,h)=X_train3(:,i);
        h=h+1;
     end
end


% data rearrange
 beta1=(data1'*data1)^(-1)*data1'*Y_train_1;
 beta2=(data2'*data2)^(-1)*data2'*Y_train_2;
 beta3=(data3'*data3)^(-1)*data3'*Y_train_3;
     
delta_test1=beta1-beta2;
delta_test2=beta1-beta3;
     
f=(abs(beta1)+abs(beta2)+abs(beta3)).*(1./(abs(delta_test1)+abs(delta_test2)));
     
[B,I]=sort(f,'descend');
     
data1=data1(:,I);
data2=data2(:,I);
data3=data3(:,I);

% Schmidt orthogonalization
[Q_hat_1,data_schmidt1]=schmidt_orthogonalization_new(data1,num_train_1);

% Schmidt orthogonalization
[Q_hat_2,data_schmidt2]=schmidt_orthogonalization_new(data2,num_train_2);

% Schmidt orthogonalization
[Q_hat_3,data_schmidt3]=schmidt_orthogonalization_new(data3,num_train_3);

% feature selection

feature_number1=size(data_schmidt1,2);

beta_ols_1=(data_schmidt1'*data_schmidt1)^(-1)*data_schmidt1'*Y_train_1;
estimated_sigma_squared_1=(Y_train_1-data_schmidt1*beta_ols_1)'*(Y_train_1-data_schmidt1*beta_ols_1)/(num_train_1-feature_number1);

beta_ols_2=(data_schmidt2'*data_schmidt2)^(-1)*data_schmidt2'*Y_train_2;
estimated_sigma_squared_2=(Y_train_2-data_schmidt2*beta_ols_2)'*(Y_train_2-data_schmidt2*beta_ols_2)/(size(data_2,1)-feature_number1);

beta_ols_3=(data_schmidt3'*data_schmidt3)^(-1)*data_schmidt3'*Y_train_3;
estimated_sigma_squared_3=(Y_train_3-data_schmidt3*beta_ols_3)'*(Y_train_3-data_schmidt3*beta_ols_3)/(size(data_3,1)-feature_number1);

delta_1=beta_ols_1-beta_ols_2;
delta_2=beta_ols_1-beta_ols_3;

for i=1:size(delta_1)
    v(i)=(12*estimated_sigma_squared_1*estimated_sigma_squared_2*estimated_sigma_squared_3)/(delta_1(i)+delta_2(i))^2;
    if v(i)>1500000
        v(i)=1500000;
    end
end

% model parameters

lambda_1=estimated_sigma_squared_2*estimated_sigma_squared_3;
lambda_2=estimated_sigma_squared_1*estimated_sigma_squared_3;
lambda_3=estimated_sigma_squared_1*estimated_sigma_squared_2;
lambda_4=v;

for i=1:size(delta_1)
    tilde_D(i)=(estimated_sigma_squared_1*estimated_sigma_squared_2*estimated_sigma_squared_3)/((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1+estimated_sigma_squared_2/num_train_2+estimated_sigma_squared_3/num_train_3);
    Q(i)=-2*tilde_D(i);
    N(i)=estimated_sigma_squared_2*num_train_1*num_train_3*((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_2/num_train_2)/((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1+estimated_sigma_squared_2/num_train_2+estimated_sigma_squared_3/num_train_3);
    M(i)=estimated_sigma_squared_1*num_train_2*num_train_3*((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1)/((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1+estimated_sigma_squared_2/num_train_2+estimated_sigma_squared_3/num_train_3);
    W(i)=estimated_sigma_squared_3*num_train_1*num_train_2*((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_3/num_train_3)/((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1+estimated_sigma_squared_2/num_train_2+estimated_sigma_squared_3/num_train_3);
    G(i)=sqrt((num_train_2*num_train_1*num_train_3)/((num_train_1*M(i)*estimated_sigma_squared_2*estimated_sigma_squared_3)+(num_train_2*N(i)*estimated_sigma_squared_1*estimated_sigma_squared_3)+(num_train_3*W(i)*estimated_sigma_squared_1*estimated_sigma_squared_2)));
    estimated_lambda_5(i)=(2*estimated_sigma_squared_1*(2-Q(i)/(sqrt(M(i)*N(i)*W(i)))))/(4*estimated_sigma_squared_1*G(i)^2);
end
lambda_5=min(estimated_lambda_5);

% Calculate the regression coefficients


[trained_beta,trained_beta_source1,trained_beta_source2]=Cp_transfer_solution_multisource_new(data_schmidt1,Y_train_1,data_schmidt2,Y_train_2,data_schmidt3,Y_train_3,lambda_1,lambda_2,lambda_3,lambda_4,lambda_5,feature_number1);

trained_beta_hat_T1=Q_hat_1*trained_beta';
trained_beta_hat_S1=Q_hat_2*trained_beta_source1';
trained_beta_hat_S2=Q_hat_3*trained_beta_source2';

% parameter estimations for TLCp criterion
feature_number=size(data1,2);

beta_1=(data1'*data1)^(-1)*data1'*Y_train_1;
estimated_sigma_squared_1=(Y_train_1-data1*beta_1)'*(Y_train_1-data1*beta_1)/(num_train_1-feature_number);

beta_2=(data2'*data2)^(-1)*data2'*Y_train_2;
estimated_sigma_squared_2=(Y_train_2-data2*beta_2)'*(Y_train_2-data2*beta_2)/(size(data_2,1)-feature_number);

beta_3=(data3'*data3)^(-1)*data3'*Y_train_3;
estimated_sigma_squared_3=(Y_train_3-data3*beta_3)'*(Y_train_3-data3*beta_3)/(size(data_3,1)-feature_number);

delta_1=beta_1-beta_2;
delta_2=beta_1-beta_3;

for i=1:size(delta_1)
    v(i)=(12*estimated_sigma_squared_1*estimated_sigma_squared_2*estimated_sigma_squared_3)/(delta_1(i)+delta_2(i))^2;
    if v(i)>1500000
        v(i)=1500000;
    end
end

tilde_lambda_1=estimated_sigma_squared_2*estimated_sigma_squared_3;
tilde_lambda_2=estimated_sigma_squared_1*estimated_sigma_squared_3;
tilde_lambda_3=estimated_sigma_squared_1*estimated_sigma_squared_2;
tilde_lambda_4=v;

for i=1:size(delta_1)
    tilde_D(i)=(estimated_sigma_squared_1*estimated_sigma_squared_2*estimated_sigma_squared_3)/((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1+estimated_sigma_squared_2/num_train_2+estimated_sigma_squared_3/num_train_3);
    Q(i)=-2*tilde_D(i);
    N(i)=estimated_sigma_squared_2*num_train_1*num_train_3*((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_2/num_train_2)/((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1+estimated_sigma_squared_2/num_train_2+estimated_sigma_squared_3/num_train_3);
    M(i)=estimated_sigma_squared_1*num_train_2*num_train_3*((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1)/((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1+estimated_sigma_squared_2/num_train_2+estimated_sigma_squared_3/num_train_3);
    W(i)=estimated_sigma_squared_3*num_train_1*num_train_2*((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_3/num_train_3)/((delta_1(i)+delta_2(i))^2+estimated_sigma_squared_1/num_train_1+estimated_sigma_squared_2/num_train_2+estimated_sigma_squared_3/num_train_3);
    G(i)=sqrt((num_train_2*num_train_1*num_train_3)/((num_train_1*M(i)*estimated_sigma_squared_2*estimated_sigma_squared_3)+(num_train_2*N(i)*estimated_sigma_squared_1*estimated_sigma_squared_3)+(num_train_3*W(i)*estimated_sigma_squared_1*estimated_sigma_squared_2)));
    estimated_lambda_5(i)=(2*estimated_sigma_squared_1*(2-Q(i)/(sqrt(M(i)*N(i)*W(i)))))/(4*estimated_sigma_squared_1*G(i)^2);
end
tilde_lambda_5=min(estimated_lambda_5);

% select threshold of u;
for h=1:feature_number
    u(h)=abs(trained_beta_hat_T1(h))/sqrt(estimated_sigma_squared_1*sum(Q_hat_1(h,:).^2)/num_train_1);
end
%The bigger the value of u1, the less features to be selected;
u_1=sort(u);
u1=[u_1,u_1(end)+1];

error_train_best=tilde_lambda_1*Y_train_1'*Y_train_1+tilde_lambda_2*Y_train_2'*Y_train_2+tilde_lambda_3*Y_train_3'*Y_train_3;
optimal_trained_beta_hat_1=zeros(feature_number,1);
for m=1:size(u1,2)
        for k=1:feature_number
            tau(k)=sqrt(estimated_sigma_squared_1*sum(Q_hat_1(k,:).^2)/num_train_1)*u1(m); % cut-off value for each feature;
            if abs(trained_beta_hat_T1(k))>=tau(k)
               trained_beta_hat_1(k)=trained_beta_hat_T1(k);
               trained_beta_hat_2(k)=trained_beta_hat_S1(k);
               trained_beta_hat_3(k)=trained_beta_hat_S2(k);
            else
                trained_beta_hat_1(k)=0;
                trained_beta_hat_2(k)=0;
                trained_beta_hat_3(k)=0;
            end
        end
        if size(trained_beta_hat_1,1)==1
           trained_beta_hat_1=trained_beta_hat_1';
        end
        if size(trained_beta_hat_2,1)==1
           trained_beta_hat_2=trained_beta_hat_2';
        end
        if size(trained_beta_hat_3,1)==1
           trained_beta_hat_3=trained_beta_hat_3';
        end
        count=sum(trained_beta_hat_1~=0);
        objective_value=tilde_lambda_1*(Y_train_1-data1*trained_beta_hat_1)'*(Y_train_1-data1*trained_beta_hat_1)+tilde_lambda_2*(Y_train_2-data2*trained_beta_hat_2)'*(Y_train_2-data2*trained_beta_hat_2)+tilde_lambda_3*(Y_train_3-data3*trained_beta_hat_3)'*(Y_train_3-data3*trained_beta_hat_3)+tilde_lambda_5*count;
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
num_data=size(data_1,1);%total number of data
Y1=data_1(:,5);
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