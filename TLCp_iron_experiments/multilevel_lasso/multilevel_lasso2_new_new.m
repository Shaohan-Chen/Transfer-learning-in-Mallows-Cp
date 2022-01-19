function [beta1,beta2]=multilevel_lasso2_new_new(data1,Y_train_1,data2,Y_train_2,lambda1,lambda2,options)

% initialize (only for two tasks)
p = size(data1, 2);  % dimensionality.
theta=ones(p,1);
gamma1=pinv(data1'*data1)*data1'*Y_train_1; % gamma
gamma2=pinv(data2'*data2)*data2'*Y_train_2;


iter_max=500;
tol=10^-3;


pre_beta1=gamma1;
pre_beta2=gamma2;

for iter=1:iter_max
    % solve for gamma
     for i=1:p
        X_11(:,i)=theta(i)*data1(:,i);
    end
    
    
    for i=1:p
        X_22(:,i)=theta(i)*data2(:,i);
    end
    
   tilde_gamma_1 = lasso(X_11,Y_train_1,'Lambda',lambda1,'Intercept',false,'Standardize',false,'reltol',10^-3);
   tilde_gamma_2 = lasso(X_22,Y_train_2,'Lambda',lambda1,'Intercept',false,'Standardize',false,'reltol',10^-3);

   gamma=[tilde_gamma_1,tilde_gamma_2];
    
   % solve for theta
   for j=1:p
       Z(:,j)=[gamma(j,1)*data1(:,j);gamma(j,2)*data2(:,j)]; 
   end

   Y=[Y_train_1;Y_train_2];
   
   theta=nng_new(Z,Y,lambda2,options);
   % update beta;
   
   beta1=theta.*gamma(:,1);
   beta2=theta.*gamma(:,2);
   
%   zhou_try=(norm(Y_train_1-data1*beta1))^2+(norm(Y_train_2-data2*beta2))^2;
   K=(norm(Y_train_1-data1*pre_beta1))^2+(norm(Y_train_2-data2*pre_beta2))^2-(norm(Y_train_1-data1*beta1))^2-(norm(Y_train_2-data2*beta2))^2;
   K=abs(K);
   if K<=tol
   break;
   else
   pre_beta1=beta1;
   pre_beta2=beta2;
   
   end
end
