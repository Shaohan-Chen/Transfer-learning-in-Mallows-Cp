function [optimal_beta_hat_T1,optimal_value]=the_original_Cp_transfer2(X,tildeX,feature_number,Y,tildeY,lambda_1,lambda_2,lambda_3,lambda_4)
% X, the target data matrix;
% tildeX, the source data matrix;
% feature_number, the dimension of the regression model;
% Y, the regression values;
% lambda 1,2,4 are the parameters in the Cp with transfer learning model.
%lambda_3 is a diagonal matrix(for simplifying the calculations);

% If no feature selected

optimal_value=lambda_1*Y'*Y+lambda_2*tildeY'*tildeY;
optimal_beta_hat_T1= zeros(feature_number,1);

for count=1:2^(feature_number)-1 % all possible choices;
   binary_representation=convert_to_binary(count,feature_number);  
   % build a matrix whose columns correspond to the nonzero indexes of binary_representation;
   j=1;
   lambda_3_prepare=eye(sum(binary_representation));
   clear X_p;
   clear tildeX_p;
   for i=1:feature_number
       
       if binary_representation(i)~=0
           X_p(:,j)=X(:,i);
           tildeX_p(:,j)=tildeX(:,i);
           lambda_3_prepare(j,j)=lambda_3(i,i);
           j=j+1;
       end
          
   end
 
   
   b_1=2*lambda_1*X_p'*Y;
   b_2=2*lambda_2*tildeX_p'*tildeY;
   C_1=2*lambda_1*X_p'*X_p;
   C_2=2*lambda_2*tildeX_p'*tildeX_p;
   row=size(X_p'*X_p);
   row=row(1);
   
   % obtain the OLS estimate for the target domain;
   beta_hat_p_T1=C_1^(-1)*b_1+C_1^(-1)*lambda_3_prepare*(2*C_2+(C_2*C_1^(-1)+eye(row))*lambda_3_prepare)^(-1)*(b_2-C_2*C_1^(-1)*b_1);
  
   % obtain the OLS estimate for the source domain;
   beta_hat_p_T2=C_2^(-1)*b_2-C_2^(-1)*lambda_3_prepare*(2*C_2+(C_2*C_1^(-1)+eye(row))*lambda_3_prepare)^(-1)*(b_2-C_2*C_1^(-1)*b_1);
  
   % the OLS estimate for the indivisual parts of the target and source domain.
   v_1=-(2*C_2+(C_2*C_1^(-1)+eye(row))*lambda_3_prepare)^(-1)*(b_2-C_2*C_1^(-1)*b_1);
   v_2=(2*C_2+(C_2*C_1^(-1)+eye(row))*lambda_3_prepare)^(-1)*(b_2-C_2*C_1^(-1)*b_1);
   
   %rebuild the k-dimentional beta_hat_T vector;
   m=1;
   for i=1:feature_number
       
       if binary_representation(i)~=0
          beta_hat_T1(i)=beta_hat_p_T1(m);
          beta_hat_T2(i)=beta_hat_p_T2(m);
           v1(i)=v_1(m);
           v2(i)=v_2(m);
           m=m+1;
       else
           beta_hat_T1(i)=0;
           beta_hat_T2(i)=0;
           v1(i)=0;
           v2(i)=0;
       end
       
          
   end
   
   % obtain the value of the objective funtion;
   objective_value=lambda_1*(Y-X*beta_hat_T1')'*(Y-X*beta_hat_T1')+lambda_2*(tildeY-tildeX*beta_hat_T2')'*(tildeY-tildeX*beta_hat_T2')+0.5*(norm(sqrt(lambda_3)*v1',2)^2+norm(sqrt(lambda_3)*v2',2)^2)+lambda_4*(sum(binary_representation));
   %objective_value=lambda_1*(Y-X*beta_hat_T1')'*(Y-X*beta_hat_T1')+lambda_2*(tildeY-tildeX*beta_hat_T2')'*(tildeY-tildeX*beta_hat_T2')+0.5*lambda_3*(sum(v_1.^2)+sum(v_2.^2))+lambda_4*(sum(binary_representation));
   %comare the obtained objective function values and find the minimizer;

   if optimal_value>objective_value
       
       optimal_value=objective_value;
       optimal_beta_hat_T1=beta_hat_T1';
   end
       
      
   
end

