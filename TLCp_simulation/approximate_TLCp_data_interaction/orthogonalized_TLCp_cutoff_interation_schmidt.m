%function [mse_original_TLCp,mse_approximate_TLCp,mse_approximate_TLCp_cutoff,mse_approximate_TLCp_original,average_original_TLCp,average_approximate_TLCp,average_approximate_TLCp_cutoff]=orthogonalized_TLCp_cutoff_interation_schmidt(num_train_1,beta)
num_train_1=20;
beta=[0.15 0.15 0.15 0.3 0.5 0]';
%beta=[0.005 0.005 0.005 0.3 0.32 1]';
%beta=[1 0.01 0.005 0.3 0.32 0.08 ]';
%beta=[1 0.32 0.3 0.08 0.01 0.005]';
% beta=[-0.255890656717379;
% -0.255890656717379;
% -0.255890656717379;
% -0.906682397000804;
% 0.732398173282907;
% 0.050514160734501;
% ];
feature_number=size(beta,1);

mse_original_TLCp=0;
mse_approximate_TLCp=0;
mse_approximate_TLCp_cutoff=0;
mse_approximate_TLCp_original=0;

num_train_2=num_train_1;  %number of sorce data
%num_train_2=40;
delta0=zeros(feature_number,1);% dissimilarity of tasks
%delta0=ones(feature_number,1)*0.1;

tic
try_number=2000;
for c=1:try_number
     [data1,data2,Y1,Y2]=build_data_simulation(num_train_1,num_train_2,feature_number,beta,delta0);
%      [data1,Y1]=build_data(num_train_1,feature_number,beta);
%      [data2,Y2]=build_data(num_train_2,feature_number,beta);
% [data1,Y1]=build_data_interation(num_train_1,feature_number,beta);
% data2=data1;
%     for l=1:num_train_2
%         e(l)=normrnd(0,1);
%     end
%     
%     Y2=data2*beta+e';

     
     beta1=(data1'*data1)^(-1)*data1'*Y1;
     beta2=(data2'*data2)^(-1)*data2'*Y2;
     
     delta_test=beta1-beta2;
     
     f=(abs(beta1)+abs(beta2)).*abs((1./delta_test));
     
     [B,I]=sort(f,'descend');
     
     data1=data1(:,I);
     data2=data2(:,I);

   
      
        [Q_hat_1,data_schmidt1]=schmidt_orthogonalization_new(data1,num_train_1);
        [Q_hat_2,data_schmidt2]=schmidt_orthogonalization_new(data2,num_train_2);
        
        %parameter estimations for approximate TLCp

         beta_ols_1=(data_schmidt1'*data_schmidt1)^(-1)*data_schmidt1'*Y1;
         estimated_sigma_squared_1=(Y1-data_schmidt1*beta_ols_1)'*(Y1-data_schmidt1*beta_ols_1)/(num_train_1-size(data_schmidt1,2));

         beta_ols_2=(data_schmidt2'*data_schmidt2)^(-1)*data_schmidt2'*Y2;
         estimated_sigma_squared_2=(Y2-data_schmidt2*beta_ols_2)'*(Y2-data_schmidt2*beta_ols_2)/(num_train_2-size(data_schmidt2,2));

         delta=beta_ols_1-beta_ols_2;

        for r=1:size(delta)
            v(r)=(4*estimated_sigma_squared_1*estimated_sigma_squared_2)/delta(r)^2;
            %v(r)=400;
        end

        % orthogonalized model parameters
  

         lambda_1=estimated_sigma_squared_2;
         lambda_2=estimated_sigma_squared_1;
         lambda_3=v;
          for i=1:size(delta)
              D_1(i)=(lambda_2*v(i))/(4*lambda_1*lambda_2*num_train_1+lambda_2*v(i)+(num_train_1/num_train_2)*lambda_1*v(i));
              D_2(i)=(lambda_1*v(i))/(4*lambda_1*lambda_2*num_train_2+lambda_1*v(i)+(num_train_2/num_train_1)*lambda_2*v(i));
              D_3(i)=(2*lambda_1*lambda_2)/(4*lambda_1*lambda_2+(1/num_train_1)*lambda_2*v(i)+(1/num_train_2)*lambda_1*v(i));
              tilde_D(i)=lambda_1*num_train_1*(D_1(i))^2+lambda_2*num_train_2*(D_2(i))^2+v(i)*(D_3(i))^2;
              Q(i)=-2*tilde_D(i);
              N(i)=-tilde_D(i)+lambda_1*num_train_1;
              M(i)=-tilde_D(i)+lambda_2*num_train_2;
              G(i)=sqrt((num_train_2*num_train_1)/((num_train_1*M(i)*estimated_sigma_squared_2)+(num_train_2*N(i)*estimated_sigma_squared_1)));
              estimated_lambda_4(i)=(2*estimated_sigma_squared_1*(2-Q(i)/(sqrt(M(i)*N(i)))))/(4*estimated_sigma_squared_1*G(i)^2);
         end
         lambda_4=min(estimated_lambda_4);
         %lambda_4=2;
      
         % Calculate the regression coefficients of target task;
         [trained_beta_target]=Cp_transfer_solution(data_schmidt1,Y1,data_schmidt2,Y2,lambda_1,lambda_2,lambda_3,lambda_4,num_train_1,num_train_2,size(data_schmidt1,2));
         trained_beta_hat_T1=Q_hat_1*trained_beta_target';
         approximate_TLCp(:,c)=trained_beta_hat_T1;

         % calculate the regression coefficints of source task;
         [trained_beta_source,tilde_v2]=Cp_transfer_source_solution(data_schmidt1,Y1,data_schmidt2,Y2,lambda_1,lambda_2,lambda_3,lambda_4,num_train_1,num_train_2,size(data_schmidt1,2));
         trained_beta_hat_T2=Q_hat_2*trained_beta_source';
          tilde_v1=-tilde_v2;
          
          % parameter estimations for TLCp criterion

          beta1=(data1'*data1)^(-1)*data1'*Y1;
          beta2=(data2'*data2)^(-1)*data2'*Y2;
     
          tilde_lambda_2=(Y1-data1*beta1)'*(Y1-data1*beta1)/(num_train_1-feature_number);
          tilde_lambda_1=(Y2-data2*beta2)'*(Y2-data2*beta2)/(num_train_2-feature_number);
          estimated_sigma_squared_1=tilde_lambda_2;
          estimated_sigma_squared_2=tilde_lambda_1;
     
          delta_original_TLCp=beta1-beta2;
     
           for i=1:feature_number
               v(i)=(4*estimated_sigma_squared_1*estimated_sigma_squared_2)/(delta_original_TLCp(i)^2);
           end

           tilde_lambda_3=diag(v);

           for i=1:feature_number
               D_1(i)=(tilde_lambda_2*v(i))/(4*tilde_lambda_1*tilde_lambda_2*num_train_1+tilde_lambda_2*v(i)+(num_train_1/num_train_2)*tilde_lambda_1*v(i));
               D_2(i)=(tilde_lambda_1*v(i))/(4*tilde_lambda_1*tilde_lambda_2*num_train_2+tilde_lambda_1*v(i)+(num_train_2/num_train_1)*tilde_lambda_2*v(i));
               D_3(i)=(2*tilde_lambda_1*tilde_lambda_2)/(4*tilde_lambda_1*tilde_lambda_2+(1/num_train_1)*tilde_lambda_2*v(i)+(1/num_train_2)*tilde_lambda_1*v(i));
               tilde_D(i)=tilde_lambda_1*num_train_1*(D_1(i))^2+tilde_lambda_2*num_train_2*(D_2(i))^2+v(i)*(D_3(i))^2;
               Q(i)=-2*tilde_D(i);
               N(i)=-tilde_D(i)+tilde_lambda_1*num_train_1;
               M(i)=-tilde_D(i)+tilde_lambda_2*num_train_2;
               G(i)=sqrt((num_train_2*num_train_1)/((num_train_1*M(i)*estimated_sigma_squared_2)+(num_train_2*N(i)*estimated_sigma_squared_1)));
               estimated_lambda_4(i)=(2*estimated_sigma_squared_1*(2-Q(i)/(sqrt(M(i)*N(i)))))/(4*estimated_sigma_squared_1*G(i)^2);
           end
          tilde_lambda_4=min(estimated_lambda_4);


        % select threshold of u;
        for h=1:feature_number
            u(h)=abs(trained_beta_hat_T1(h))/sqrt(estimated_sigma_squared_1*sum(Q_hat_1(h,:).^2)/num_train_1);
        end
        u_1=sort(u);
        u1=[u_1,u_1(end)+1];
       error_train_best=tilde_lambda_1*Y1'*Y1+tilde_lambda_2*Y2'*Y2;
       optimal_trained_beta_hat_1=zeros(feature_number,1);
      for m=1:size(u1,2)
          for k=1:feature_number
            tau(k)=sqrt(estimated_sigma_squared_1*sum(Q_hat_1(k,:).^2)/num_train_1)*u1(m); % cut-off value for each feature;
            if abs(trained_beta_hat_T1(k))>=tau(k)
               trained_beta_hat_1(k)=trained_beta_hat_T1(k);
               trained_beta_hat_2(k)=trained_beta_hat_T2(k);
            else
                trained_beta_hat_1(k)=0;
                trained_beta_hat_2(k)=0;
            end
          end
          if size(trained_beta_hat_1,1)==1
           trained_beta_hat_1=trained_beta_hat_1';
          end
          if size(trained_beta_hat_2,1)==1
             trained_beta_hat_2=trained_beta_hat_2';
          end
          v1=(trained_beta_hat_1-trained_beta_hat_2)/2;
          v2=(trained_beta_hat_2-trained_beta_hat_1)/2;
         count=sum(trained_beta_hat_1~=0);
         %objective_value=tilde_lambda_1*(Y1-data1*trained_beta_hat_1)'*(Y1-data1*trained_beta_hat_1)+tilde_lambda_2*(Y2-data2*trained_beta_hat_2)'*(Y2-data2*trained_beta_hat_2)+0.5*(norm(sqrt(tilde_lambda_3)*v1,2)^2+norm(sqrt(tilde_lambda_3)*v2,2)^2)+tilde_lambda_4*count;
         objective_value=tilde_lambda_1*(Y1-data1*trained_beta_hat_1)'*(Y1-data1*trained_beta_hat_1)+tilde_lambda_2*(Y2-data2*trained_beta_hat_2)'*(Y2-data2*trained_beta_hat_2)+tilde_lambda_4*count;
         error_train=objective_value;
         if error_train<error_train_best
            error_train_best=error_train;
            u_best=u1(m);
            optimal_trained_beta_hat_1=trained_beta_hat_1;
             
         end
      end

     approximate_TLCp_cutoff(:,c)=optimal_trained_beta_hat_1;% approximate TLCp;
%      
%      approximate_TLCp_cutoff(:,c)=zeros(feature_number,1);
%      approximate_TLCp(:,c)=zeros(feature_number,1);
     
     % original TLCp (for dissimilarity=0 case)
     
     tilde_lambda_2=1;
     tilde_lambda_1=1;
 
     estimated_sigma_squared_1=1;
     estimated_sigma_squared_2=1;
     
     for i=1:feature_number
         v1(i)=100000000;
     end
     tilde_lambda_3=diag(v1);
     tilde_lambda_4=2;
     
     [original_TLCp(:,c),optimal_value]=the_original_Cp_transfer2(data1,data2,feature_number,Y1,Y2,tilde_lambda_1,tilde_lambda_2,tilde_lambda_3,tilde_lambda_4);
  %   original_TLCp(:,c)=zeros(feature_number,1);
     % estimated beta rearrange
     for a=1:feature_number
         approximate_TLCp2(a,1)=approximate_TLCp(find(I==a),c);
         approximate_TLCp_cutoff2(a,1)=approximate_TLCp_cutoff(find(I==a),c);
         original_TLCp2(a,1)=original_TLCp(find(I==a),c);
     end
     
     approximate_TLCp(:,c)=approximate_TLCp2;
     approximate_TLCp_cutoff(:,c)=approximate_TLCp_cutoff2;
     original_TLCp(:,c)=original_TLCp2;
     
     
     %error analysis
        
        approximate_TLCp1(1,c)=(approximate_TLCp(1,c)+approximate_TLCp(2,c)+approximate_TLCp(3,c));
        approximate_TLCp_cutoff1(1,c)=(approximate_TLCp_cutoff(1,c)+approximate_TLCp_cutoff(2,c)+approximate_TLCp_cutoff(3,c));
        original_TLCp1(1,c)=(original_TLCp(1,c)+original_TLCp(2,c)+original_TLCp(3,c));
        beta_true(1,1)=beta(1)+beta(2)+beta(3);
       
        for l=2:feature_number-2
            approximate_TLCp1(l,c)=approximate_TLCp(l+2,c);
            approximate_TLCp_cutoff1(l,c)=approximate_TLCp_cutoff(l+2,c);
            original_TLCp1(l,c)=original_TLCp(l+2,c);
            beta_true(l,1)=beta(l+2);
        end
        
    %record mse value each time
    
    mse_approximate_TLCp_0(c)=(approximate_TLCp1(:,c)-beta_true)'*(approximate_TLCp1(:,c)-beta_true);
    mse_approximate_TLCp_cutoff_0(c)=(approximate_TLCp_cutoff1(:,c)-beta_true)'*(approximate_TLCp_cutoff1(:,c)-beta_true);
    mse_original_TLCp_0(c)=(original_TLCp1(:,c)-beta_true)'*(original_TLCp1(:,c)-beta_true);
    
    mse_approximate_TLCp=mse_approximate_TLCp+(approximate_TLCp1(:,c)-beta_true)'*(approximate_TLCp1(:,c)-beta_true); 
    mse_approximate_TLCp_cutoff=mse_approximate_TLCp_cutoff+(approximate_TLCp_cutoff1(:,c)-beta_true)'*(approximate_TLCp_cutoff1(:,c)-beta_true);
    mse_original_TLCp=mse_original_TLCp+(original_TLCp1(:,c)-beta_true)'*(original_TLCp1(:,c)-beta_true); 
    mse_approximate_TLCp_original=mse_approximate_TLCp_original+(approximate_TLCp1(:,c)-original_TLCp1(:,c))'*(approximate_TLCp1(:,c)-original_TLCp1(:,c)); 
    
    
%     beta_true=beta;
%     mse_approximate_TLCp=mse_approximate_TLCp+(approximate_TLCp(:,c)-beta_true)'*(approximate_TLCp(:,c)-beta_true); 
%     mse_approximate_TLCp_cutoff=mse_approximate_TLCp_cutoff+(approximate_TLCp_cutoff(:,c)-beta_true)'*(approximate_TLCp_cutoff(:,c)-beta_true);
%     mse_original_TLCp=mse_original_TLCp+(original_TLCp(:,c)-beta_true)'*(original_TLCp(:,c)-beta_true); 
%     mse_approximate_TLCp_original=mse_approximate_TLCp_original+(approximate_TLCp(:,c)-original_TLCp(:,c))'*(approximate_TLCp(:,c)-original_TLCp(:,c)); 
end
 
   mse_original_TLCp=mse_original_TLCp/try_number;
   mse_approximate_TLCp=mse_approximate_TLCp/try_number;
   mse_approximate_TLCp_cutoff=mse_approximate_TLCp_cutoff/try_number;
   mse_approximate_TLCp_original=mse_approximate_TLCp_original/try_number;
   toc
%    average_approximate_TLCp=mean(approximate_TLCp');
%    average_approximate_TLCp_cutoff=mean(approximate_TLCp_cutoff');
%    average_original_TLCp=mean(original_TLCp');
   
%    p_o=zeros(feature_number-2,1);
%    p=zeros(feature_number-2,1);
%    p_cutoff=zeros(feature_number-2,1);
%    
%    for j=1:try_number
%           if original_TLCp(1,j)~=0 || original_TLCp(2,j)~=0 || original_TLCp(3,j)~=0
%              p_o(1)=p_o(1)+1;
%           end
%           
%    end     
%    for i=2:feature_number-2
%       for j=1:try_number
%           if original_TLCp(i+2,j)~=0
%             p_o(i)=p_o(i)+1;
%           end
%        
%       end
%    end
%    
% 
%       for j=1:try_number
%           if approximate_TLCp(1,j)~=0 || approximate_TLCp(2,j)~=0 || approximate_TLCp(3,j)~=0
%              p(1)=p(1)+1;
%           end
%           
%       end     
%    for i=2:feature_number-2
%       for j=1:try_number
%           if approximate_TLCp(i+2,j)~=0
%             p(i)=p(i)+1;
%           end
%        
%       end
%    end
%    
%     for j=1:try_number
%           if approximate_TLCp_cutoff(1,j)~=0 || approximate_TLCp_cutoff(2,j)~=0 || approximate_TLCp_cutoff(3,j)~=0
%              p_cutoff(1)=p_cutoff(1)+1;
%           end
%           
%     end     
%    for i=2:feature_number-2
%       for j=1:try_number
%           if approximate_TLCp_cutoff(i+2,j)~=0
%             p_cutoff(i)=p_cutoff(i)+1;
%           end
%        
%       end
%    end
%    p_o=p_o/try_number;
%    p=p/try_number;
%    p_cutoff=p_cutoff/try_number;
%    
%    for j=1:try_number
%           if approximate_TLCp_cutoff(1,j)>0.001 || approximate_TLCp_cutoff(2,j)>0.001 || approximate_TLCp_cutoff(3,j)>0.001
%              p_cutoff(1)=p_cutoff(1)+1;
%           end
%           
%     end     
%    for i=2:feature_number-2
%       for j=1:try_number
%           if approximate_TLCp_cutoff(i+2,j)>0.0010
%             p_cutoff(i)=p_cutoff(i)+1;
%           end
%        
%       end
%    end