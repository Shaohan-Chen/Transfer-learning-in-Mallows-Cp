%function [explained_variance]=iron_multitask_L21_new(num_train_1)
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
N=size(data1,1);
optimal_value=1000000;

 for lambda=[10^-6 10^-5 10^-4 0.001 0.01 0.1 1 10 100 1000]%0.2:0.2:2
 sse=0;    
     for i=1:5
         indice=crossvalind('Kfold',N,5);
         
         test=(indice==i);
         train=~test;
         
         X=cell(1,2);
         X{1}=data1(train,:);
         X{2}=data2;
         Y=cell(1,2);
         Y{1}=Y_train_1(train);
         Y{2}=Y_train_2;
     
         [W funcVal] = Least_L21(X, Y, lambda);
         
         
         sse=sse+sum((data1(test,:)*W(:,1)-Y_train_1(test)).^2);
         
     
     end
     
     CVerr=sse/5;
     
     if CVerr<optimal_value
         optimal_value=CVerr;
         optimal_lambda=lambda;
     end
 end
 
% transformed into cell

X=cell(1,2);
X{1}=data1;
X{2}=data2;

Y=cell(1,2);
Y{1}=Y_train_1;
Y{2}=Y_train_2;

[W funcVal] = Least_L21(X, Y, optimal_lambda);

%% error analysis
coef=W(:,1);
coef0=0;
num_data=size(data_1,1);%total number of data
Y1=data_1(:,5);
total_variance=sum((Y1-mean(Y1)).^2)/num_data; % for total data
error=(Y_test_1-((data11*coef+coef0).*std(Y_1)+mean(Y_1)))'*(Y_test_1-((data11*coef+coef0).*std(Y_1)+mean(Y_1)));% only for test data
error_test=error/num_test_1;
explained_variance=1-error_test/total_variance;
toc