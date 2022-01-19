function [X,Y]=data_cell(num_train,num_train_2)

num_test=100;

data1=xlsread('baogang_test2.xlsx');
data2=xlsread('laigang_test2.xlsx');

% target data

size_data=size(data1);   % cancel data without complete information
j=1;
for i=1:size_data(1)
    if  min(abs(data1(i,:))) ~= 0
        data1_selected(j,:)=data1(i,:);
       j=j+1;
   end
end

%rearrange the order of the data
r=randperm(size(data1_selected,1));
data_1_selected=data1_selected(r,:);

Y_train_1=data_1_selected(1:num_train,1);
X_train_1=[ones(num_train,1),data_1_selected(1:num_train,2:end)];
%data1=[ones(num_train_1,1),data_1_selected(1:num_train_1,2:end)];

X_test_1=data_1_selected(num_train+1:num_train+num_test,:);
Y_test_1=data_1_selected(num_train+1:num_train+num_test,1);
%data11=[ones(num_test,1),X_test_1(1:num_test,2:end)];% target domain test data

% source data

%X_train_2=data2(1:num_train_2,2:end);
Y_train_2=data2(1:num_train_2,1);
X_train_2=[ones(num_train_2,1),data2(1:num_train_2,2:end)];
%data2=[ones(num_train_2,1),data_2(1:num_train_2,2:end)];

% transformed into cell

X=cell(1,2);
X{1}=X_train_1;
X{2}=X_train_2;

Y=cell(1,2);
Y{1}=Y_train_1;
Y{2}=Y_train_2;




