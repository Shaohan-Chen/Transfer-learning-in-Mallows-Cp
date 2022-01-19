function [hatbeta]=Cp_solution(X,Y,lambda_p,data_size, feature_number)

for i=1:feature_number
    
    if lambda_p<data_size*(1/data_size*X(:,i)'*Y)^2
        
       hatbeta(i)=1/data_size*X(:,i)'*Y;
       
    else
        
        hatbeta(i)=0;
        
    end

end

