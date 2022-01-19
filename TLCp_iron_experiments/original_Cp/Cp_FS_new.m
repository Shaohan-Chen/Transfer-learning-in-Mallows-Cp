%function [explained_variance]=Cp_FS_new(num_train)
%tic
clear
num_train=210;
num_test=100;

% data preparation
data=xlsread('furnace_A.xlsx');

size_data=size(data);   % cancel data without complete information
j=1;
for i=1:size_data(1)
    if  min(abs(data(i,:))) ~= 0
        data_1(j,:)=data(i,:);
       j=j+1;
   end
end

%rearrange the order of the data
r=randperm(size(data_1,1));
data_1=data_1(r,:);

% training set
X_1=data_1(1:num_train,2:end);
Y_1=data_1(1:num_train,1);

% training data standardized
X_1_normalization=(X_1-mean(X_1))./(std(X_1));
Y_1_normalization=(Y_1-mean(Y_1))/(std(Y_1));
X_train=[X_1_normalization];
data1=X_train;
Y_train=Y_1_normalization;

%test data
X_2=data_1(num_train+1:num_train+num_test,2:end);
Y_2=data_1(num_train+1:num_train+num_test,1);

%test data standardized
X_2_normalization=(X_2-mean(X_1))./(std(X_1));
X_test=[X_2_normalization];
data11=X_test;
Y_test=Y_2;

%% feature selection
k=size(data1,2); % number of features
beta_ols=(data1'*data1)^(-1)*data1'*Y_train;
estimated_sigma_squared=(Y_train-data1*beta_ols)'*(Y_train-data1*beta_ols)/(num_train-k);
lambda=2*estimated_sigma_squared; % model parameter

% Calculate the regression coefficients
[trained_beta_hat,optimal_value]=the_original_Cp(data1,k,Y_train,lambda);

%%error calculation
num_data=size(data_1,1);%total number of data
Y1=data_1(:,1);
total_variance=sum((Y1-mean(Y1)).^2)/num_data; % for total data
error=(Y_test-(data11*trained_beta_hat.*std(Y_1)+mean(Y_1)))'*(Y_test-(data11*trained_beta_hat.*std(Y_1)+mean(Y_1)));% only for test data
error_test=error/num_test;
explained_variance=1-error_test/total_variance;
%toc
