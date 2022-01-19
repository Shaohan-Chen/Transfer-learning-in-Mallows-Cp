function [explained_variance]=Cp_FS_new(num_train)
num_test=30;

% data preparation
data=load('parkinsons_updrs copy.txt');
% using the data of the first patient as the target data;
j=1;
for i=1:size(data,1)

    if data(i,1)==1
       data_1(j,:)=data(i,:);
       j=j+1;
   end
end

%rearrange the order of the data
r=randperm(size(data_1,1));
data_1=data_1(r,:);

%training data

X_1=data_1(1:num_train,[4,7:end]);
Y_1=data_1(1:num_train,5);

% standardize training data
X_1_normalization=(X_1-mean(X_1))./std(X_1);
Y_1_normalization=(Y_1-mean(Y_1))/(std(Y_1));
%X_train=[ones(num_train,1),X_1_normalization];
X_train=[X_1_normalization];
Y_train=Y_1_normalization;

%test data
X_2=data_1(num_train+1:num_train+num_test,[4,7:end]);
Y_2=data_1(num_train+1:num_train+num_test,5);

%test data standardized
X_2_normalization=(X_2-mean(X_1))./(std(X_1));
% X_test=[ones(num_test,1),X_2_normalization];
X_test=[X_2_normalization];
Y_test=Y_2;

% delete some correlated features

data1(:,1)=X_train(:,1); 
data11(:,1)=X_test(:,1);

h=2;
for i=2:size(X_train,2)
     A=[X_train(:,i),data1];
     if rank(A)==h
        data1(:,h)=X_train(:,i); 
        data11(:,h)=X_test(:,i);
        h=h+1;
     end
end
%% feature selection
k=size(data1,2); % number of features
beta_ols=(data1'*data1)^(-1)*data1'*Y_train;
estimated_sigma_squared=(Y_train-data1*beta_ols)'*(Y_train-data1*beta_ols)/(num_train-k);
lambda=2*estimated_sigma_squared; % model parameter

% Calculate the regression coefficients
[trained_beta_hat,optimal_value]=the_original_Cp(data1,k,Y_train,lambda);
beta0=mean(Y_train-data1*trained_beta_hat);
%% error calculation
coef=trained_beta_hat;
num_data=size(data_1,1);%total number of data
Y1=data_1(:,5);
total_variance=sum((Y1-mean(Y1)).^2)/num_data; % for total data
error=(Y_test-((data11*coef+beta0).*std(Y_1)+mean(Y_1)))'*(Y_test-((data11*coef+beta0).*std(Y_1)+mean(Y_1)));% only for test data
error_test=error/num_test;
explained_variance=1-error_test/total_variance;