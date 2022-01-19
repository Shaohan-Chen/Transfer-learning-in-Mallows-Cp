%function [explained_variance]=TLCp_FS_test(num_train_1)

clear
tic
num_train_1=170;
num_test_1=30;

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

% % training data standardized
for i=1:size(X_1,2)
    
    if  i==4 || i==5
        X_1_normalization(:,i)=(X_1(:,i)-mean(X_1(:,i)))/(std(X_1(:,i)));
    else
        X_1_normalization(:,i)=X_1(:,i);
    end
end

X_train1=[ones(num_train_1,1),X_1_normalization];
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

X_test_1=[ones(num_test_1,1),X_2_normalization];
Y_test_1=Y_2;

% source domain data
num_train_2=size(data_2,1);
X_train_2=data_2(1:num_train_2,2:end-1);
Y_train_2=data_2(1:num_train_2,1);

% source data standardized
for i=1:size(X_2,2)
    
    if  (i==4 || i==5) && std(X_train_2(:,i))~=0
        X_2_source_normalization(:,i)=(X_train_2(:,i)-mean(X_train_2(:,i)))/(std(X_train_2(:,i)));
    else
        X_2_source_normalization(:,i)=X_train_2(:,i);
    end
end

X_train2=[ones(num_train_2,1),X_2_source_normalization];
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
% % delete ones
% for t=2:size(data1,2)
%     tilde_data1(:,t-1)=data1(:,t);
%     tilde_data2(:,t-1)=data2(:,t);
%     tilde_data11(:,t-1)=data11(:,t);
%     
% end
% 
% data1=tilde_data1;
% data2=tilde_data2;
% data11=tilde_data11;

%% feature selection
feature_number=size(data1,2);
beta_ols_1=(data1'*data1)^(-1)*data1'*Y_train_1;
estimated_sigma_squared_1=(Y_train_1-data1*beta_ols_1)'*(Y_train_1-data1*beta_ols_1)/(num_train_1-feature_number);

beta_ols_2=(data2'*data2)^(-1)*data2'*Y_train_2;
estimated_sigma_squared_2=(Y_train_2-data2*beta_ols_2)'*(Y_train_2-data2*beta_ols_2)/(num_train_2-feature_number);

delta=beta_ols_1-beta_ols_2;

for i=1:size(delta)
    v(i)=(4*estimated_sigma_squared_1*estimated_sigma_squared_2)/delta(i)^2;
end

% model parameters

lambda_1=estimated_sigma_squared_2;
lambda_2=estimated_sigma_squared_1;
lambda_3=diag(v);
%lambda_4=2*estimated_sigma_squared_1; % corresponding to the target data
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

% Calculate the regression coefficients

[trained_beta_hat_T1,optimal_value]=the_original_Cp_transfer2(data1,data2,feature_number,Y_train_1,Y_train_2,lambda_1,lambda_2,lambda_3,lambda_4);
beta0=mean(Y_train_1-data1*trained_beta_hat_T1);
%% error calculation
num_data=size(data_1,1);%total number of data
Y1=data_1(:,1);
total_variance=sum((Y1-mean(Y1)).^2)/num_data; % for total data
error=(Y_test_1-((data11*trained_beta_hat_T1+beta0).*std(Y_1)+mean(Y_1)))'*(Y_test_1-((data11*trained_beta_hat_T1+beta0).*std(Y_1)+mean(Y_1)));% only for test data
error_test=error/num_test_1;
explained_variance=1-error_test/total_variance;
toc
