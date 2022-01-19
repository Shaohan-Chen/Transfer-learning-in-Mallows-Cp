function [data1,data2,Y1,Y2]=build_data_simulation(num_train_target,num_train_source,feature_number,beta,delta)

  for j=1:num_train_target
         for k=1:feature_number-2
             data_target(j,k)=normrnd(0,1);
         end
  end
  
  for j=1:num_train_source
         for k=1:feature_number-2
             data_source(j,k)=normrnd(0,1);
         end
  end
  % copy the first column of data twice and add small random noise to the first two columns;
  
    for l=1:num_train_target
        e1(l)=normrnd(0,1);
    end
    
   for h=1:num_train_target
        e2(h)=normrnd(0,1);
   end
   
   for a=1:num_train_target
        e3(a)=normrnd(0,1);
   end
   
   for g=1:num_train_target
        e(g)=normrnd(0,1);
   end
    

   data1=[data_target(:,1)+e1'./1000 data_target(:,1)+e2'./1000 data_target(:,1)+e3'./1000 data_target(:,2:end)];
     
    Y1=data1*beta+e';
    
    % copy the first column of data twice and add small random noise to the first two columns;
  
    for l=1:num_train_source
        d1(l)=normrnd(0,1);
    end
    
   for h=1:num_train_source
        d2(h)=normrnd(0,1);
   end
   
   for a=1:num_train_source
        d3(a)=normrnd(0,1);
   end
   
   for g=1:num_train_source
        d(g)=normrnd(0,1);
   end
    

   data2=[data_source(:,1)+d1'./1000 data_source(:,1)+d2'./1000 data_source(:,1)+d3'./1000 data_source(:,2:end)];
     
    Y2=data2*(beta+delta)+d';
    
    
    
    