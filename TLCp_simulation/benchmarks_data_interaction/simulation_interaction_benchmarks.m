%function [mse_aggregate_original_Cp,mse_aggregate_lasso,mse_aggregate_stepwise,mse_aggregate_univariate]=simulation_interaction_benchmarks(num_train_1,beta)
num_train_1=20;
%beta=[0.005 0.005 0.005 0.3 0.32 1]';
%beta=[0.005 0.005 0.005 0.32 0 0]';
%beta=[1 0.01 0.005 0.3 0.32 0.08 ]';
beta=[0.15 0.15 0.15 0.3 0.5 0]';
feature_number=size(beta,1);

mse_aggregate_original_Cp=0;
mse_aggregate_lasso=0;
mse_aggregate_stepwise=0;
mse_aggregate_univariate=0;

delta0=zeros(feature_number,1);% dissimilarity of tasks
%delta0=ones(feature_number,1)*0.1;


num_train_2=num_train_1;  %number of sorce data
tic
try_number=1;
for c=1:try_number
      [data1,data2,Y1,Y2]=build_data_simulation(num_train_1,num_train_2,feature_number,beta,delta0);
      %[data1,Y1]=build_data(num_train_1,feature_number,beta);
      %[data2,Y2]=build_data(num_train_2,feature_number,beta);
%         [data1,data2,Y1,Y2]=build_data_simulation(num_train_1,num_train_2,feature_number,beta,delta0);
%        data2=data1;
%       for g=1:num_train_1
%           d(g)=normrnd(0,1);
%       end
%       Y2=data2*beta+d';
      % aggragate data
      
      %data=[data1;data2];
      %Y=[Y1;Y2];
  %    num_train=size(Y,1);
      data=data1;
      Y=Y1;
      num_train=size(Y,1);
      
      %% aggregate original Cp
%       
%       %parameter estimation for original Cp
%       
%       
%        
%        beta_ols_o=(data'*data)^(-1)*data'*Y;
%        estimated_sigma_squared_o=(Y-data*beta_ols_o)'*(Y-data*beta_ols_o)/(num_train-size(data,2));
%        lambda_o=2*estimated_sigma_squared_o;
%        
%        % original Cp;
%        [original_Cp(:,c),optimal_value]=the_original_Cp(data,feature_number,Y,lambda_o);
       
       %% aggregate lasso
       
%         [b,fitinfo] = lasso(data,Y,'CV',5);
%        idxLambdaMinMSE = find(fitinfo.MSE==min(fitinfo.MSE));
%         lasso0(:,c) = b(:,idxLambdaMinMSE);
%        
       %% aggregate stepwise
%        
       [b1,se,pval,inmodel,stats,nextstep,history] = stepwisefit(data,Y,'Display','off');
       stepwise(:,c)=b1.*inmodel';
       
%        %% aggragate univariate
       
%        [r,pval]=corr(data,Y,'type','pearson');
%        j=1;
%        for i=1:size(pval,1)
%          if pval(i)<0.05
%             X_selected(:,j)=data(:,i);
%             j=j+1;
%          end
%        end
% 
%  if j==1
%     beta_selected=zeros(feature_number,1);
% else
%     beta_selected=(X_selected'*X_selected)^(-1)*X_selected'*Y;
% end
% 
% % dimension recover
% 
% m=1;
%    for i=1:size(pval,1)
%        
%        if pval(i)<0.05
%           trained_beta_hat(i)=beta_selected(m);
%            m=m+1;
%        else
%            trained_beta_hat(i)=0;
%        end
%        
%          
%    end
%    
%    if j==1
%        trained_beta_hat=zeros(1,feature_number);
%    end
%  
%     
%    univariate(:,c)=trained_beta_hat';
%    
   
original_Cp(:,c)=zeros(feature_number,1);
lasso0(:,c)=zeros(feature_number,1);
%stepwise(:,c)=zeros(feature_number,1);
univariate(:,c)=zeros(feature_number,1);
   
   %% error analysis
   original_Cp1(1,c)=(original_Cp(1,c)+original_Cp(2,c)+original_Cp(3,c));
   lasso1(1,c)=(lasso0(1,c)+lasso0(2,c)+lasso0(3,c));
   stepwise1(1,c)=(stepwise(1,c)+stepwise(2,c)+stepwise(3,c));
   univariate1(1,c)=(univariate(1,c)+univariate(2,c)+univariate(3,c));
   beta_true(1,1)=beta(1)+beta(2)+beta(3);
       
        for l=2:feature_number-2
            original_Cp1(l,c)=original_Cp(l+2,c);
            lasso1(l,c)=lasso0(l+2,c);
            stepwise1(l,c)=stepwise(l+2,c);
            univariate1(l,c)=univariate(l+2,c);
            beta_true(l,1)=beta(l+2);
        end
        
   % record each mse value each time
   
%     mse_aggregate_original_Cp_0(c)=(original_Cp1(:,c)-beta_true)'*(original_Cp1(:,c)-beta_true); 
%     mse_aggregate_lasso_0(c)=(lasso1(:,c)-beta_true)'*(lasso1(:,c)-beta_true); 
%     mse_aggregate_stepwise_0(c)=(stepwise1(:,c)-beta_true)'*(stepwise1(:,c)-beta_true); 
%     mse_aggregate_univariate_0(c)=(univariate1(:,c)-beta_true)'*(univariate1(:,c)-beta_true); 
    
    mse_original_Cp_0(c)=(original_Cp1(:,c)-beta_true)'*(original_Cp1(:,c)-beta_true); 
    mse_lasso_0(c)=(lasso1(:,c)-beta_true)'*(lasso1(:,c)-beta_true); 
    mse_stepwise_0(c)=(stepwise1(:,c)-beta_true)'*(stepwise1(:,c)-beta_true); 
    mse_univariate_0(c)=(univariate1(:,c)-beta_true)'*(univariate1(:,c)-beta_true); 
        
    mse_aggregate_original_Cp=mse_aggregate_original_Cp+(original_Cp1(:,c)-beta_true)'*(original_Cp1(:,c)-beta_true); 
    mse_aggregate_lasso=mse_aggregate_lasso+(lasso1(:,c)-beta_true)'*(lasso1(:,c)-beta_true);
    mse_aggregate_stepwise=mse_aggregate_stepwise+(stepwise1(:,c)-beta_true)'*(stepwise1(:,c)-beta_true); 
    mse_aggregate_univariate=mse_aggregate_univariate+(univariate1(:,c)-beta_true)'*(univariate1(:,c)-beta_true); 
    
    
%     beta_true=beta;
%     mse_aggregate_original_Cp=mse_aggregate_original_Cp+(original_Cp(:,c)-beta_true)'*(original_Cp(:,c)-beta_true); 
%     mse_aggregate_lasso=mse_aggregate_lasso+(lasso0(:,c)-beta_true)'*(lasso0(:,c)-beta_true);
%     mse_aggregate_stepwise=mse_aggregate_stepwise+(stepwise(:,c)-beta_true)'*(stepwise(:,c)-beta_true); 
%     mse_aggregate_univariate=mse_aggregate_univariate+(univariate(:,c)-beta_true)'*(univariate(:,c)-beta_true); 
   
      
end

mse_aggregate_original_Cp=mse_aggregate_original_Cp/try_number;
mse_aggregate_lasso=mse_aggregate_lasso/try_number;
mse_aggregate_stepwise=mse_aggregate_stepwise/try_number;
mse_aggregate_univariate=mse_aggregate_univariate/try_number;
toc