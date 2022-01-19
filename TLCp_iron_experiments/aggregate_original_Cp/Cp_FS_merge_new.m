%function [explained_variance]=Cp_FS_merge_new(num_train_1)
clear
num_train_1=210;
num_test_1=100;
% data preparation
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
X_train_1=[X_1_normalization];
data1=X_train_1;
Y_train_1=Y_1_normalization;

%test data
X_2=data_11(num_train_1+1:num_train_1+num_test_1,2:end);
Y_2=data_11(num_train_1+1:num_train_1+num_test_1,1);

%test data standardized
X_2_normalization=(X_2-mean(X_1))./(std(X_1));
X_test_1=[X_2_normalization];
data11=X_test_1;
Y_test_1=Y_2;

% source domain data
data_2=xlsread('furnace_B.xlsx');
num_train_2=size(data_2,1);
X_train_2=data_2(1:num_train_2,2:end);
Y_train_2=data_2(1:num_train_2,1);

% data standardized
X_train_2_normalization=(X_train_2-mean(X_train_2))./std(X_train_2);
Y_train_2=(Y_train_2-mean(Y_train_2))/(std(Y_train_2)); 
data2=[X_train_2_normalization];

% merged data
X_train=[data1;data2];
Y_train=[Y_train_1;Y_train_2];

%% feature selection
k=size(data1,2); % number of features
beta_ols=(X_train'*X_train)^(-1)*X_train'*Y_train;
estimated_sigma_squared=(Y_train-X_train*beta_ols)'*(Y_train-X_train*beta_ols)/(size(X_train,1)-k);
lambda=2*estimated_sigma_squared; % model parameter

% Calculate the regression coefficients
[trained_beta_hat,optimal_value]=the_original_Cp(X_train,k,Y_train,lambda);

%% error calculation
num_data=size(data_11,1);%total number of data
Y1=data_11(:,1);
total_variance=sum((Y1-mean(Y1)).^2)/num_data; % for total data
error=(Y_test_1-(data11*trained_beta_hat.*std(Y_1)+mean(Y_1)))'*(Y_test_1-(data11*trained_beta_hat.*std(Y_1)+mean(Y_1)));% only for test data
error_test=error/num_test_1;
explained_variance=1-error_test/total_variance;
