function [optimal_beta_hat_T1,optimal_value]=the_original_Cp_transfer3(X1,Y1,X2,Y2,X3,Y3,lambda_1,lambda_2,lambda_3,lambda_4,lambda_5,feature_number)

% If no feature selected

optimal_value=lambda_1*Y1'*Y1+lambda_2*Y2'*Y2+lambda_3*Y3'*Y3;
optimal_beta_hat_T1= zeros(feature_number,1);

for count=1:2^(feature_number)-1 % all possible choices;
   binary_representation=convert_to_binary(count,feature_number);  
   % build a matrix whose columns correspond to the nonzero indexes of binary_representation;
   j=1;
   k=sum(binary_representation);
   lambda_4_prepare=eye(sum(binary_representation));
   clear X_p1;
   clear X_p2;
   clear X_p3;
   for i=1:feature_number
       
       if binary_representation(i)~=0
           X_p1(:,j)=X1(:,i);
           X_p2(:,j)=X2(:,i);
           X_p3(:,j)=X3(:,i);
           lambda_4_prepare(j,j)=lambda_4(i,i);
           j=j+1;
       end
          
   end
 
  % obtain the ols estimations for the selected dimentions;
  
  [alpha_0,v_1,v_2,v_3]=the_original_TLCp_multisource_linearequations(X_p1,Y1,X_p2,Y2,X_p3,Y3,lambda_1,lambda_2,lambda_3,lambda_4_prepare,k);
   
   beta_hat_p_T1=alpha_0+v_1;
   beta_hat_p_T2=alpha_0+v_2;
   beta_hat_p_T3=alpha_0+v_3;
   %rebuild the k-dimentional beta_hat_T vector;
   m=1;
   for i=1:feature_number
       
       if binary_representation(i)~=0
          beta_hat_T1(i)=beta_hat_p_T1(m);
          beta_hat_T2(i)=beta_hat_p_T2(m);
          beta_hat_T3(i)=beta_hat_p_T3(m);
           v1(i)=v_1(m);
           v2(i)=v_2(m);
           v3(i)=v_3(m);
           m=m+1;
       else
           beta_hat_T1(i)=0;
           beta_hat_T2(i)=0;
           beta_hat_T3(i)=0;
           v1(i)=0;
           v2(i)=0;
           v3(i)=0;
       end
       
          
   end
   
   % obtain the value of the objective funtion;
   
   objective_value=lambda_1*(Y1-X1*beta_hat_T1')'*(Y1-X1*beta_hat_T1')+lambda_2*(Y2-X2*beta_hat_T2')'*(Y2-X2*beta_hat_T2')+lambda_3*(Y3-X3*beta_hat_T3')'*(Y3-X3*beta_hat_T3')+0.5*(norm(sqrt(lambda_4)*v1',2)^2+norm(sqrt(lambda_4)*v2',2)^2+norm(sqrt(lambda_4)*v3',2)^2)+lambda_5*(sum(binary_representation));
   
   %comare the obtained objective function values and find the minimizer;

   if optimal_value>objective_value
       
       optimal_value=objective_value;
       optimal_beta_hat_T1=beta_hat_T1';
   end
       
      
   
end