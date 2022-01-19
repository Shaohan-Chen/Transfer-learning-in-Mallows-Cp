clear;
%beta=[0.24 0.01 0.005 0.3 0.32 0.08 0 0.26 0.25 0]';
%beta=rand(20,1);
beta=[0.42 0.89 0.96 0.20 0 0.65 0.84 0 0.29 0]';
delta0=[-0.4 0.8 -1 0.20 0 -0.7 0.8 0 0.3 0]';
%delta0=[0.24 -0.01 0.005 -0.3 -0.32 0.08 0 -0.26 -0.25 0]';

feature_number=size(beta,1);
max_iter=20000;
tildedata_size=20;

index=1;
for data_size=[20:20:180]
    
    X=prepare_data(feature_number,data_size);
    tildeX=prepare_data(feature_number,tildedata_size);
    count_1=[0,1,10,30,50,70,90,100,200,300,400];
    for count=1:size(count_1,2)
        count_2=count_1(count);
        delta=delta0*count_2/100;
        norm_delta(count)=norm(delta,2);
        chosen_probability_transfer=zeros(1,feature_number);
   
        for j=1:max_iter
            correctly_chosen_transfer_Cp(j)=0;
            [Y]=preparedata21(data_size,X,beta); 
            [tildeY]=preparedata22(tildedata_size,delta,tildeX,beta);
           %transfer + Cp
           %randomly choose parameters;
           lambda_1=2;
           lambda_2=3;
           lambda_3=[1 2 3 4 5 6 7 8 9 10];
           lambda_4=1;
           
           %lambda_1=1;
           %estimated_sigma_squared_1=1;
           %estimated_sigma_squared_2=1;
           %lambda_2=1;
           %for i=1:feature_number
            %   if delta(i)==0
            %       v(i)=10^6;
            %   else
             %  v(i)=(4*estimated_sigma_squared_1*estimated_sigma_squared_2)/delta(i)^2;
             %  end
           %end
           %lambda_3=v;
           
           num_train_1=data_size;
           num_train_2=tildedata_size;
            %for i=1:feature_number
                
             %  D_1(i)=(lambda_2*v(i))/(4*lambda_1*lambda_2*num_train_1+lambda_2*v(i)+(num_train_1/num_train_2)*lambda_1*v(i));
               % D_2(i)=(lambda_1*v(i))/(4*lambda_1*lambda_2*num_train_2+lambda_1*v(i)+(num_train_2/num_train_1)*lambda_2*v(i));
               % D_3(i)=(2*lambda_1*lambda_2)/(4*lambda_1*lambda_2+(1/num_train_1)*lambda_2*v(i)+(1/num_train_2)*lambda_1*v(i));
               % tilde_D(i)=lambda_1*num_train_1*(D_1(i))^2+lambda_2*num_train_2*(D_2(i))^2+v(i)*(D_3(i))^2;
                %Q(i)=-2*tilde_D(i);
                %N(i)=-tilde_D(i)+lambda_1*num_train_1;
                %M(i)=-tilde_D(i)+lambda_2*num_train_2;
                %G(i)=sqrt((num_train_2*num_train_1)/((num_train_1*M(i)*estimated_sigma_squared_2)+(num_train_2*N(i)*estimated_sigma_squared_1)));
                %estimated_lambda_4(i)=(2*estimated_sigma_squared_1*(2-Q(i)/(sqrt(M(i)*N(i)))))/(4*estimated_sigma_squared_1*G(i)^2);
               
            %end
            
           %lambda_4=min(estimated_lambda_4);
           

      hatbeta_transfer(j,:)=Cp_transfer_solution(X,Y,tildeX,tildeY,lambda_1,lambda_2,lambda_3,lambda_4,data_size,tildedata_size,feature_number);
      
      
   
   
       for k=1:feature_number
          if (hatbeta_transfer(j,k)~=0 && beta(k)~=0)|| (hatbeta_transfer(j,k)==0 && beta(k)==0)
              correctly_chosen_transfer_Cp(j)=correctly_chosen_transfer_Cp(j)+1;
          end
           
       end
       
        end
   
   % calculate the total number of correctly selected variables
    correctly_transfer_Cp(count)=0;
    for j=1:max_iter
         correctly_transfer_Cp(count)=correctly_transfer_Cp(count)+correctly_chosen_transfer_Cp(j);
    end
   
   mse_transfer(count)=0;
   for i=1:max_iter
      mse_transfer(count)=mse_transfer(count)+(hatbeta_transfer(i,:)'-beta)'*(hatbeta_transfer(i,:)'-beta);
   end
   
   mse_transfer1(count)=mse_transfer(count)/max_iter;
   correctly_variables_transfer_Cp(count)=correctly_transfer_Cp(count)/max_iter;
   
 end
    
    correctly_selected_variables_transfer_Cp(index,:)=correctly_variables_transfer_Cp;
    mse_transfer_Cp(index,:)=mse_transfer1;
    

% Cp part;
    
    for h=1:max_iter  
        correctly_chosen_Cp(h)=0;
       [Y]=preparedata21(data_size,X,beta);   
       
        hatbeta_pure(h,:)=Cp_solution(X,Y,2,data_size,feature_number);
        
        for k=1:feature_number
          if (hatbeta_pure(h,k)~=0 && beta(k)~=0) || (hatbeta_pure(h,k)==0 && beta(k)==0)
              correctly_chosen_Cp(h)=correctly_chosen_Cp(h)+1;
          end
           
       end
    end
    % calculate the total number of correctly selected variables
    correctly_Cp(index)=0;
    for j=1:max_iter
         correctly_Cp(index)=correctly_Cp(index)+correctly_chosen_Cp(j);
    end
    
    mse_Cp(index)=0;
    for j=1:max_iter
         mse_Cp(index)=mse_Cp(index)+(hatbeta_pure(j,:)'-beta)'*(hatbeta_pure(j,:)'-beta);
    end
    
    % transfer data analysis
   correctly_variables_Cp(index,:)=(correctly_Cp(index)/max_iter)*ones(1,11);
   mse_org_Cp(index,:)=(mse_Cp(index)/max_iter)*ones(1,11);
   index=index+1;
end

