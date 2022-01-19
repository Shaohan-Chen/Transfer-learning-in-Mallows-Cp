function [optimal_beta_hat,optimal_value]=the_original_Cp(X,k,Y,lambda)
% X, the data matrix;
% k, the dimension of the regression model;
% Y, the regression values;
% lambda, the parameter in the Cp model.

% If no feature selected

optimal_value=Y'*Y;
optimal_beta_hat= zeros(k,1);

for count=1:2^k-1 % all possible choices;
   binary_representation=convert_to_binary(count,k);  
   % build a matrix whose columns correspond to the nonzero indexes of binary_representation;
   j=1;
   clear X_p;
   for i=1:k
       
       if binary_representation(i)~=0
           X_p(:,j)=X(:,i);
           j=j+1;
       end
           
   end
   
   % obtain the OLS estimate;
   beta_hat_p=(X_p'*X_p)^(-1)*X_p'*Y;
   %rebuild the k-dimentional beta_hat vector;
   m=1;
   for i=1:k
       
       if binary_representation(i)~=0
          beta_hat(i)=beta_hat_p(m);
           m=m+1;
       else
           beta_hat(i)=0;
       end
       
         
   end
   % obtain the value of the objective funtion for Cp;
   objective_value=(Y-X*beta_hat')'*(Y-X*beta_hat')+lambda*(sum(binary_representation));
   
   %comare the obtained objective function values and find the minimizer;

   if optimal_value>objective_value
       
       optimal_value=objective_value;
       optimal_beta_hat=beta_hat';
   end
       
      
   
end


