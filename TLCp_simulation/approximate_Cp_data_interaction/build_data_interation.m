function [data1,Y]=build_data_interation(num_train,feature_number,beta)

  for j=1:num_train
         for k=1:feature_number-2
             data(j,k)=normrnd(0,1);
         end
  end
  
  % copy the first column of data twice and add small random noise to the first two columns;
  
    for l=1:num_train
        e1(l)=normrnd(0,1);
    end
    
   for h=1:num_train
        e2(h)=normrnd(0,1);
   end
   
   for a=1:num_train
        e3(a)=normrnd(0,1);
   end
   
   for g=1:num_train
        e(g)=normrnd(0,1);
   end
    
   % data1=[data(:,1)+e1'./num_train^2 data(:,2)+e2'./num_train^2 data];
   data1=[data(:,1)+e1'./1000 data(:,1)+e2'./1000 data(:,1)+e3'./1000 data(:,2:end)];
     
    Y=data1*beta+e';
    
    
    
    