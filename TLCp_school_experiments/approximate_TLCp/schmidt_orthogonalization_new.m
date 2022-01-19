function [Q_hat,data_schmidt]=schmidt_orthogonalization_new(data,num_train)

[row,col]=size(data);
%data_schmidt_1=zeros(row,col);
coefficient=zeros(col,col);
data_schmidt_1(:,1)=data(:,1);
coefficient(1,1)=1;


for i=2:col
    coefficient(i,i)=1;
    for j=1:i-1
        if norm(data_schmidt_1(:,j))<0.00001*num_train %0.00000001*num_train
            coefficient(j,i)=0;
            data(:,i)=data(:,i)-coefficient(j,i)*data_schmidt_1(:,j);
        else
            coefficient(j,i)=(data_schmidt_1(:,j)'*data(:,i))/(data_schmidt_1(:,j)'*data_schmidt_1(:,j));
            data(:,i)=data(:,i)-coefficient(j,i)*data_schmidt_1(:,j);
        end
    end
    if norm(data(:,i))<0.00001*num_train
       data_schmidt_1(:,i)=zeros(num_train,1); 
    else
       data_schmidt_1(:,i)=data(:,i); 
    end
end

Q_tilde=inv(coefficient);

%normalization
for i=1:col
       
      if norm(data_schmidt_1(:,i))>0
  
          data_schmidt(:,i)=sqrt(num_train)*data_schmidt_1(:,i)/norm(data_schmidt_1(:,i));
      else
          
          data_schmidt(:,i)=sqrt(num_train)*data_schmidt_1(:,i);
      end
   
end


for i=1:size(coefficient,1)
    
    if norm(data_schmidt_1(:,i))>0
    
         Q_hat(:,i)=sqrt(num_train)*Q_tilde(:,i)/norm(data_schmidt_1(:,i));
    else
         Q_hat(:,i)=Q_tilde(:,i);
    end
       
    
end
    
%cutoff zeros
j=1;
for i=1:col
       
      if norm(data_schmidt(:,i))>0
  
          data_schmidt2(:,j)=data_schmidt(:,i);
          Q_hat2(:,j)=Q_hat(:,i);
          j=j+1;
          
      end
   
end

Q_hat=Q_hat2;
data_schmidt=data_schmidt2;
