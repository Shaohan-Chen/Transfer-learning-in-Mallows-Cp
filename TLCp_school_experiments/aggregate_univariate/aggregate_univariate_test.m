%function [explained_variance]=aggregate_univariate_test(num_train_1)
clear
tic
num_test_1=30;
num_train_1=170;


% data preparation
data_X=load('school feature.mat');
data_Y=load('school response.mat');

data_train_1=data_X.X{1,1};
data_train_2=data_X.X{1,18};

Y_train_1=data_Y.Y{1,1};
Y_train_2=data_Y.Y{1,18};

data_1=[Y_train_1,data_train_1];
data_2=[Y_train_2,data_train_2];

%rearrange the order of the data
r=randperm(size(data_1,1));
data_1=data_1(r,:);

%training data

X_1=data_1(1:num_train_1,2:end-1);
Y_1=data_1(1:num_train_1,1);

% training data standardized
for i=1:size(X_1,2)
    
    if  i==4 || i==5
        X_1_normalization(:,i)=(X_1(:,i)-mean(X_1(:,i)))/(std(X_1(:,i)));
    else
        X_1_normalization(:,i)=X_1(:,i);
    end
end

X_train1=[ones(size(X_1,1),1),X_1_normalization];
%X_train1=[X_1_normalization];
Y_train_1=(Y_1-mean(Y_1))/std(Y_1);

% testing data
X_2=data_1(num_train_1+1:num_train_1+num_test_1,2:end-1);
Y_2=data_1(num_train_1+1:num_train_1+num_test_1,1);

%test data standardized
for i=1:size(X_2,2)
    
    if  i==4 || i==5
        X_2_normalization(:,i)=(X_2(:,i)-mean(X_1(:,i)))/(std(X_1(:,i)));
    else
        X_2_normalization(:,i)=X_2(:,i);
    end
end

X_test_1=[ones(size(X_2,1),1),X_2_normalization];
%X_test_1=[X_2_normalization];
Y_test_1=Y_2;

% source domain data
num_train_2=size(data_2,1);
X_train_2=data_2(1:num_train_2,2:end-1);
Y_train_2=data_2(1:num_train_2,1);

%source data standardized
for i=1:size(X_train_2,2)
    
    if  (i==4 || i==5) && std(X_train_2(:,i))~=0
        X_2_source_normalization(:,i)=(X_train_2(:,i)-mean(X_train_2(:,i)))/(std(X_train_2(:,i)));
    else
        X_2_source_normalization(:,i)=X_train_2(:,i);
    end
end

X_train2=[ones(num_train_2,1),X_2_source_normalization];
%X_train2=[X_2_source_normalization];
Y_train_2=(Y_train_2-mean(Y_train_2))/std(Y_train_2);

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
% delete ones
for t=2:size(data1,2)
    tilde_data1(:,t-1)=data1(:,t);
    tilde_data2(:,t-1)=data2(:,t);
    tilde_data11(:,t-1)=data11(:,t);
    
end

data1=tilde_data1;
data2=tilde_data2;
data11=tilde_data11;

%merge data
train_data=[data1;data2];
train_Y=[Y_train_1;Y_train_2];

%% feature selection
feature_number=size(data1,2);
[r,pval]=corr(train_data,train_Y,'type','Pearson');
j=1;
for i=1:size(pval,1)
    if pval(i)<0.05
        X_selected(:,j)=train_data(:,i);
        j=j+1;
    end
end

if j==1
    beta_selected=zeros(feature_number,1);
else
    beta_selected=(X_selected'*X_selected)^(-1)*X_selected'*train_Y;
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
 
 beta0=mean(train_Y-train_data*trained_beta_hat');

%% error calculation
coef=trained_beta_hat';
coef0=beta0;
num_data=size(data_1,1);%total number of data
Y1=data_1(:,1);
total_variance=sum((Y1-mean(Y1)).^2)/num_data; % for total data
error=(Y_test_1-((data11*coef+coef0).*std(Y_1)+mean(Y_1)))'*(Y_test_1-((data11*coef+coef0).*std(Y_1)+mean(Y_1)));% only for test data
error_test=error/num_test_1;
explained_variance=1-error_test/total_variance;
toc