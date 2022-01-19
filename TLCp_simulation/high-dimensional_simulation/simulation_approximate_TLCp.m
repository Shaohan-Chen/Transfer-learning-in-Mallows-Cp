function [mse4,mse4_cutoff]=simulation_approximate_TLCp(num_train_1,feature_number,data1,beta,delta0,d,Q_hat1,data_schmidt1)
D=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,250,300];
B=[D(1),D(2),D(3),D(5),D(10),D(22)];
delta=delta0*B(d)/100;   
% solve approximate TLCp problem;
[Y1,Y2]=build_data2(num_train_1,data1,beta,delta);
          num_train_2=num_train_1;
          lambda_1=1;
          lambda_2=1; 
          tilde_delta=pinv(Q_hat1)*delta;
        
            for h=1:size(tilde_delta,1)
                if tilde_delta(h)==0
                    v(h)=1000000;
                else
                    v(h)=4/(tilde_delta(h)^2);
                end
            end
          lambda_3=v;
          estimated_sigma_squared_1=1;
          estimated_sigma_squared_2=1;
            for k=1:size(tilde_delta)
                D_1(k)=(lambda_2*v(k))/(4*lambda_1*lambda_2*num_train_1+lambda_2*v(k)+(num_train_1/num_train_2)*lambda_1*v(k));
                D_2(k)=(lambda_1*v(k))/(4*lambda_1*lambda_2*num_train_2+lambda_1*v(k)+(num_train_2/num_train_1)*lambda_2*v(k));
                D_3(k)=(2*lambda_1*lambda_2)/(4*lambda_1*lambda_2+(1/num_train_1)*lambda_2*v(k)+(1/num_train_2)*lambda_1*v(k));
                tilde_D(k)=lambda_1*num_train_1*(D_1(k))^2+lambda_2*num_train_2*(D_2(k))^2+v(k)*(D_3(k))^2;
                Q(k)=-2*tilde_D(k);
                N(k)=-tilde_D(k)+lambda_1*num_train_1;
                M(k)=-tilde_D(k)+lambda_2*num_train_2;
                G(k)=sqrt((num_train_2*num_train_1)/((num_train_1*M(k)*estimated_sigma_squared_2)+(num_train_2*N(k)*estimated_sigma_squared_1)));
                estimated_lambda_4(k)=(2*estimated_sigma_squared_1*(2-Q(k)/(sqrt(M(k)*N(k)))))/(4*estimated_sigma_squared_1*G(k)^2);
            end
            
         lambda_4=min(estimated_lambda_4);
         data_schmidt2=data_schmidt1;
         Q_hat2=Q_hat1;
         q1=Cp_transfer_solution(data_schmidt1,Y1,data_schmidt2,Y2,lambda_1,lambda_2,lambda_3,lambda_4,num_train_1,num_train_2,size(data_schmidt1,2));
         q=Q_hat1*(q1)';
         transformed_hatbeta2=q;
        
        [trained_beta_source,v2]=Cp_transfer_source_solution(data_schmidt1,Y1,data_schmidt2,Y2,lambda_1,lambda_2,lambda_3,lambda_4,num_train_1,num_train_2,size(data_schmidt1,2));
        trained_beta_hat_T2=Q_hat2*trained_beta_source';
        
         % parameters for original TLCp;
         
          for h=1:size(delta,1)
                if delta(h)==0
                    v(h)=1000000;
                else
                    v(h)=4/(delta(h)^2);
                end
          end
          tilde_lambda_3=v;
           for k=1:size(delta)
                D_1(k)=(lambda_2*v(k))/(4*lambda_1*lambda_2*num_train_1+lambda_2*v(k)+(num_train_1/num_train_2)*lambda_1*v(k));
                D_2(k)=(lambda_1*v(k))/(4*lambda_1*lambda_2*num_train_2+lambda_1*v(k)+(num_train_2/num_train_1)*lambda_2*v(k));
                D_3(k)=(2*lambda_1*lambda_2)/(4*lambda_1*lambda_2+(1/num_train_1)*lambda_2*v(k)+(1/num_train_2)*lambda_1*v(k));
                tilde_D(k)=lambda_1*num_train_1*(D_1(k))^2+lambda_2*num_train_2*(D_2(k))^2+v(k)*(D_3(k))^2;
                Q(k)=-2*tilde_D(k);
                N(k)=-tilde_D(k)+lambda_1*num_train_1;
                M(k)=-tilde_D(k)+lambda_2*num_train_2;
                G(k)=sqrt((num_train_2*num_train_1)/((num_train_1*M(k)*estimated_sigma_squared_2)+(num_train_2*N(k)*estimated_sigma_squared_1)));
                estimated_lambda_4(k)=(2*estimated_sigma_squared_1*(2-Q(k)/(sqrt(M(k)*N(k)))))/(4*estimated_sigma_squared_1*G(k)^2);
            end
            
         tilde_lambda_4=min(estimated_lambda_4);
         
        % select threshold of u for approximate TLCp;

            for h=1:feature_number
                tilde_u(h)=abs(q(h))/sqrt(estimated_sigma_squared_1*sum(Q_hat1(h,:).^2)/num_train_1);
            end

            data2=data1;
            optimal_trained_beta_hat_2=q;
            optimal_trained_beta_hat_2_source=trained_beta_hat_T2;
            error_train_best=lambda_1*(Y1-data1*q)'*(Y1-data1*q)+lambda_2*(Y2-data2*trained_beta_hat_T2)'*(Y2-data2*trained_beta_hat_T2)+lambda_4*feature_number;
            error_residual_1=Y1-data1*optimal_trained_beta_hat_2;
            error_residual_2=Y2-data2*optimal_trained_beta_hat_2_source;
            trained_beta_hat_2=q;
            trained_beta_hat_2_source=trained_beta_hat_T2;
            [tilde_a,tilde_b]=sort(tilde_u);
            count2=feature_number;
           for m=1:feature_number 
               beta_hat_2_selected=trained_beta_hat_2(tilde_b(m));
               beta_hat_2_selected_source=trained_beta_hat_2_source(tilde_b(m));
               trained_beta_hat_2(tilde_b(m))=0;
               trained_beta_hat_2_source(tilde_b(m))=0;
                if size(trained_beta_hat_2,1)==1
                   trained_beta_hat_2=trained_beta_hat_2';
                end
                if size(trained_beta_hat_2_source,1)==1
                   trained_beta_hat_2_source=trained_beta_hat_2_source';
                end
            %count2=sum(trained_beta_hat_2~=0);
            count2=count2-1;
            %objective_value=lambda_1*(Y1-data1*trained_beta_hat_2)'*(Y1-data1*trained_beta_hat_2)+lambda_2*(Y2-data2*trained_beta_hat_2_source)'*(Y2-data2*trained_beta_hat_2_source)+tilde_lambda_4*count2;
            error_residual_1=error_residual_1+data1(:,tilde_b(m))*beta_hat_2_selected;
            error_residual_2=error_residual_2+data2(:,tilde_b(m))*beta_hat_2_selected_source;
            objective_value=sum(error_residual_1.^2)+sum(error_residual_2.^2)+tilde_lambda_4*count2;
            error_train=objective_value;
            if error_train<error_train_best
               error_train_best=error_train;
               optimal_trained_beta_hat_2=trained_beta_hat_2;
               
            end
          end

        transformed_hatbeta_2=optimal_trained_beta_hat_2;
        
        mse4=(transformed_hatbeta2-beta)'*(transformed_hatbeta2-beta);
        mse4_cutoff=(transformed_hatbeta_2-beta)'*(transformed_hatbeta_2-beta);
