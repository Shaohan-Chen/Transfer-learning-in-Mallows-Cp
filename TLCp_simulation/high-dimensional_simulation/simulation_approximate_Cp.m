function [mse2,mse2_cutoff]=simulation_approximate_Cp(num_train_1,feature_number,data1,beta,Q_hat1,data_schmidt1)
%solve approximate Cp problem;
delta=zeros(feature_number,1);
[Y1,Y2]=build_data2(num_train_1,data1,beta,delta);
p1=Cp_solution(data_schmidt1,Y1,2,num_train_1,size(data_schmidt1,2));
p=Q_hat1*(p1)';
transformed_hatbeta1=p;
%parameter settings
estimated_sigma_squared=1;
lambda=2;
%feature selection 
for h=1:feature_number
    u(h)=abs(p(h))/sqrt(estimated_sigma_squared*sum(Q_hat1(h,:).^2)/num_train_1);
end
optimal_trained_beta_hat_1=p;
error_train_best=(Y1-data1*optimal_trained_beta_hat_1)'*(Y1-data1*optimal_trained_beta_hat_1)+lambda*feature_number;        
error_residual=Y1-data1*optimal_trained_beta_hat_1;
trained_beta_hat_1=p;
[a,b]=sort(u);
count1=feature_number;
   for m=1:size(u,2)
        beta_hat_1_selected=trained_beta_hat_1(b(m));
        trained_beta_hat_1(b(m))=0;
   
       if size(trained_beta_hat_1,1)==1
          trained_beta_hat_1=trained_beta_hat_1';
       end
          %count1=sum(trained_beta_hat_1~=0);
          count1=count1-1;
          %error_train=(Y1-data1*trained_beta_hat_1)'*(Y1-data1*trained_beta_hat_1)+lambda*count1; 
          %error_train=sum((Y1-data1*trained_beta_hat_1).^2)+lambda*count1;
          error_residual=error_residual+data1(:,b(m))*beta_hat_1_selected;
          error_train=sum(error_residual.^2)+lambda*count1;
          if error_train<error_train_best
             error_train_best=error_train;
             optimal_trained_beta_hat_1=trained_beta_hat_1;
          end
    end

       transformed_hatbeta_1=optimal_trained_beta_hat_1;
       
  mse2=(transformed_hatbeta1-beta)'*(transformed_hatbeta1-beta);
  mse2_cutoff=(transformed_hatbeta_1-beta)'*(transformed_hatbeta_1-beta);      
       