%function [explained_variance]=aggregate_univariate(num_train_1)
clear;
tic
num_train_1=290;
num_test_1=100;

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
% merged data
X_train=[data1;data2];
Y_train=[Y_train_1;Y_train_2];

%% feature selection
feature_number=size(data1,2);
[r,pval]=corr(X_train,Y_train,'type','Pearson');
j=1;
for i=1:size(pval,1)
    if pval(i)<0.05
        X_selected(:,j)=X_train(:,i);
        j=j+1;
    end
end

if j==1
    beta_selected=zeros(feature_number,1);
else
    beta_selected=(X_selected'*X_selected)^(-1)*X_selected'*Y_train;
end

% dimension recover

m=1;
   for i=1:size(pval,1)
       
       if pval(i)<0.05
          trained_beta_hat(i)=beta_selected(m);
           m=m+1;
       else
           trained_beta_hat(i)=0;
       end
       
         
   end
   
   if j==1
       trained_beta_hat=zeros(1,feature_number);
   end

%intercept term
 
 beta0=mean(Y_train-X_train*trained_beta_hat');

%% error calculation
coef=trained_beta_hat';
coef0=beta0;
num_data=size(data_11,1);%total number of data
Y1=data_11(:,1);
total_variance=sum((Y1-mean(Y1)).^2)/num_data; % for total data
error=(Y_test_1-((data11*coef+coef0).*std(Y_1)+mean(Y_1)))'*(Y_test_1-((data11*coef+coef0).*std(Y_1)+mean(Y_1)));% only for test data
error_test=error/num_test_1;
explained_variance=1-error_test/total_variance;
toc
