%function [mse_original_Cp,mse_approximate_Cp,mse_approximate_Cp_cutoff,mse_approximate_Cp_original,average_original_Cp,average_approximate_Cp,average_approximate_Cp_cutoff]=orthogonalized_Cp_cutoff_interation_schmidt(num_train,beta)
clear;
num_train=20;
%num_train=20;
%beta=[1 0.01 0.005 0.3 0.32 0.08 ]';
%beta=[1 0 0 0.3 0.5 0]';
%beta=[0.15 0.15 0.15 0.3 0.5 0]';
%beta=[0.005 0.005 0.005 0.4 0.5 1 ]';
%beta=[-0.8 -0.4 -0.8 0.01 0.01 0.1]';
beta=[-0.255890656717379;
-0.255890656717379;
-0.255890656717379;
-0.906682397000804;
0.732398173282907;
0.050514160734501;
];
feature_number=size(beta,1);

mse_original_Cp=0;
mse_approximate_Cp=0;
mse_approximate_Cp_cutoff=0;
mse_approximate_Cp_original=0;
try_number=2000;
for i=1:try_number
        [data,Y]=build_data_interation(num_train,feature_number,beta);
        
        % schmidt orthogonalization
        
        [Q_hat,data_schmidt]=schmidt_orthogonalization_new(data,num_train);
        
        %parameter estimation for approximate Cp
        
        beta_ols=(data_schmidt'*data_schmidt)^(-1)*data_schmidt'*Y;
        estimated_sigma_squared=(Y-data_schmidt*beta_ols)'*(Y-data_schmidt*beta_ols)/(num_train-size(data_schmidt,2));
        lambda=2*estimated_sigma_squared;
        
       feature_number1=size(data_schmidt,2);
       hatbeta_orthogonalization=Cp_solution(data_schmidt,Y,lambda,num_train,feature_number1);
       transformed_hatbeta(:,i)=Q_hat*hatbeta_orthogonalization'; % approximate Cp;
       
       %parameter estimation for original Cp
       
       beta_ols_o=(data'*data)^(-1)*data'*Y;
       estimated_sigma_squared_o=(Y-data*beta_ols_o)'*(Y-data*beta_ols_o)/(num_train-size(data,2));
       lambda_o=2*estimated_sigma_squared_o;
        % lambda_o=2;
       
       % original Cp;
       [hatbeta(:,i),optimal_value]=the_original_Cp(data,feature_number,Y,lambda_o);
  
       % select threshold of u
      for h=1:feature_number
          u(h)=abs(transformed_hatbeta(h,i))/sqrt(estimated_sigma_squared*sum(Q_hat(h,:).^2)/num_train);
      end
      u_1=sort(u);
      u1=[u_1,u_1(end)+1];
      error_train_best=Y'*Y;
for m=1:size(u1,2)
        for k=1:feature_number
            tau(k)=sqrt(estimated_sigma_squared*sum(Q_hat(k,:).^2)/num_train)*u1(m); % cut-off value for each feature;
            if abs(transformed_hatbeta(k,i))>=tau(k)
               trained_beta_hat_1(k)=transformed_hatbeta(k,i);
            else
                trained_beta_hat_1(k)=0;
            end
        end
        if size(trained_beta_hat_1,1)==1
           trained_beta_hat_1=trained_beta_hat_1';
        end
        count=sum(trained_beta_hat_1~=0);
        lambda=2*estimated_sigma_squared;
        error_train=(Y-data*trained_beta_hat_1)'*(Y-data*trained_beta_hat_1)+lambda*count; 
       if error_train<error_train_best
          error_train_best=error_train;
          u_best=u1(m);
          optimal_trained_beta_hat_1=trained_beta_hat_1;
       end
           
end

        transformed_hatbeta_1(:,i)=optimal_trained_beta_hat_1; % approximate Cp cutoff;
        
        %error analysis
        
        hatbeta1(1,i)=(hatbeta(1,i)+hatbeta(2,i)+hatbeta(3,i));
        transformed_hatbeta1(1,i)=(transformed_hatbeta(1,i)+transformed_hatbeta(2,i)+transformed_hatbeta(3,i));
        transformed_hatbeta1_1(1,i)=(transformed_hatbeta_1(1,i)+transformed_hatbeta_1(2,i)+transformed_hatbeta_1(3,i));
        beta_true(1,1)=beta(1)+beta(2)+beta(3);
       
        for l=2:feature_number-2
            hatbeta1(l,i)=hatbeta(l+2,i);
            transformed_hatbeta1(l,i)=transformed_hatbeta(l+2,i);
            transformed_hatbeta1_1(l,i)=transformed_hatbeta_1(l+2,i);
            beta_true(l,1)=beta(l+2);
        end
        
        mse_original_Cp=mse_original_Cp+(hatbeta1(:,i)-beta_true)'*(hatbeta1(:,i)-beta_true);
        mse_approximate_Cp=mse_approximate_Cp+(transformed_hatbeta1(:,i)-beta_true)'*(transformed_hatbeta1(:,i)-beta_true);
        mse_approximate_Cp_cutoff=mse_approximate_Cp_cutoff+(transformed_hatbeta1_1(:,i)-beta_true)'*(transformed_hatbeta1_1(:,i)-beta_true);
        mse_approximate_Cp_original=mse_approximate_Cp_original+(transformed_hatbeta1(:,i)-hatbeta1(:,i))'*(transformed_hatbeta1(:,i)-hatbeta1(:,i));
end
   mse_original_Cp=mse_original_Cp/try_number;
   mse_approximate_Cp=mse_approximate_Cp/try_number;
   mse_approximate_Cp_cutoff=mse_approximate_Cp_cutoff/try_number;
   mse_approximate_Cp_original=mse_approximate_Cp_original/try_number;
   average_approximate_Cp=mean(transformed_hatbeta');
   average_approximate_Cp_cutoff=mean(transformed_hatbeta_1');
   average_original_Cp=mean(hatbeta');
   
   
   p_o=zeros(feature_number-2,1);
   p=zeros(feature_number-2,1);
   p_cutoff=zeros(feature_number-2,1);
   
   for j=1:try_number
          if hatbeta(1,j)~=0 || hatbeta(2,j)~=0 || hatbeta(3,j)~=0
             p_o(1)=p_o(1)+1;
          end
          
      end     
   for i=2:feature_number-2
      for j=1:try_number
          if hatbeta(i+2,j)~=0
            p_o(i)=p_o(i)+1;
          end
       
      end
   end

      for j=1:try_number
          if transformed_hatbeta(1,j)~=0 || transformed_hatbeta(2,j)~=0 || transformed_hatbeta(3,j)~=0
             p(1)=p(1)+1;
          end
          
      end     
   for i=2:feature_number-2
      for j=1:try_number
          if transformed_hatbeta(i+2,j)~=0
            p(i)=p(i)+1;
          end
       
      end
   end
   
    for j=1:try_number
          if transformed_hatbeta_1(1,j)~=0 || transformed_hatbeta_1(2,j)~=0 || transformed_hatbeta_1(3,j)~=0
             p_cutoff(1)=p_cutoff(1)+1;
          end
          
    end     
   for i=2:feature_number-2
      for j=1:try_number
          if transformed_hatbeta_1(i+2,j)~=0
            p_cutoff(i)=p_cutoff(i)+1;
          end
       
      end
   end
   
   p_o=p_o/try_number;
   p=p/try_number;
   p_cutoff=p_cutoff/try_number;