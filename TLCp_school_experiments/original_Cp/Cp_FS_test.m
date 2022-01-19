%function [explained_variance]=Cp_FS_test(num_train)
clear
tic
num_train=170;
num_test=30;

% data preparation
data_X=load('school feature.mat');
data_Y=load('school response.mat');

data_train_1=data_X.X{1,1};

Y_train_1=data_Y.Y{1,1};

data=[Y_train_1,data_train_1];

%rearrange the order of the data
r=randperm(size(data,1));
data_1=data(r,:);

%training data

X_1=data_1(1:num_train,2:end-1);
Y_1=data_1(1:num_train,1);

% training data standardized
for i=1:size(X_1,2)
    
    if  i==4 || i==5
        X_1_normalization(:,i)=(X_1(:,i)-mean(X_1(:,i)))/(std(X_1(:,i)));
    else
        X_1_normalization(:,i)=X_1(:,i);
    end
end


X_train=[ones(num_train,1),X_1_normalization];
Y_train=(Y_1-mean(Y_1))/std(Y_1);

% testing data
X_2=data_1(num_train+1:num_train+num_test,2:end-1);
Y_2=data_1(num_train+1:num_train+num_test,1);

% %test data standardized
for i=1:size(X_2,2)
    
    if  i==4 || i==5
        X_2_normalization(:,i)=(X_2(:,i)-mean(X_1(:,i)))/(std(X_1(:,i)));
    else
        X_2_normalization(:,i)=X_2(:,i);
    end
end

X_test=[ones(num_test,1),X_2_normalization];
Y_test=Y_2;

% delete some correlated features

num_feature=size(X_train,2);%original feature numbers

data1(:,1)=X_train(:,1); 
data11(:,1)=X_test(:,1);

h=2;
for i=2:num_feature
     A=[X_train(:,i),data1];
     if rank(A)==h
        data1(:,h)=X_train(:,i); % training data
        data11(:,h)=X_test(:,i); % testing data
        h=h+1;
     end
end
% % delete ones
% for t=2:size(data1,2)
%     tilde_data1(:,t-1)=data1(:,t);
%     tilde_data11(:,t-1)=data11(:,t);
%     
% end
% 
% data1=tilde_data1;
% data11=tilde_data11;

%% feature selection
k=size(data1,2); % number of features
beta_ols=(data1'*data1)^(-1)*data1'*Y_train;
estimated_sigma_squared=(Y_train-data1*beta_ols)'*(Y_train-data1*beta_ols)/(num_train-k);
lambda=2*estimated_sigma_squared; % model parameter

% Calculate the regression coefficients
[trained_beta_hat,optimal_value]=the_original_Cp(data1,k,Y_train,lambda);
coef0=0;%mean(Y_train-data1*trained_beta_hat);
%% error calculation
coef=trained_beta_hat;
num_data=size(data_1,1);%total number of data
Y1=data_1(:,1);
total_variance=sum((Y1-mean(Y1)).^2)/num_data; % for total data
error=(Y_test-((data11*coef+coef0).*std(Y_1)+mean(Y_1)))'*(Y_test-((data11*coef+coef0).*std(Y_1)+mean(Y_1)));% only for test data
error_test=error/num_test;
explained_variance=1-error_test/total_variance;
toc
