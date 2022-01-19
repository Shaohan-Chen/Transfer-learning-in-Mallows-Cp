function [hatbeta_T]=Cp_transfer_solution(X,Y,tildeX,tildeY,lambda_1,lambda_2,lambda_3,lambda_4,data_size_target,data_size_source,feature_number)

for i=1:feature_number
    
    D_1(i)=(lambda_2*lambda_3(i)*data_size_source)/(4*lambda_1*lambda_2*data_size_source*data_size_target+data_size_source*lambda_2*lambda_3(i)+data_size_target*lambda_1*lambda_3(i));
    D_3(i)=2*lambda_1*data_size_target*D_1(i)/lambda_3(i);
    Z(i)=1/data_size_target*X(:,i)'*Y;
    H(i)=1/data_size_source*tildeX(:,i)'*tildeY;
    
    tildeD(i)=0.5*lambda_3(i)*D_3(i);
    
    if lambda_4+(tildeD(i)-lambda_2*data_size_source)*H(i)^2+(tildeD(i)-lambda_1*data_size_target)*Z(i)^2-2*tildeD(i)*H(i)*Z(i)<0
        
        hatbeta_T(i)=(1-D_1(i))*1/data_size_target*X(:,i)'*Y+D_1(i)*1/data_size_source*tildeX(:,i)'*tildeY;
    
    else
        hatbeta_T(i)=0;
    end
end
