
function [data1]=build_data1(num_train_1,feature_number)

for j=1:num_train_1
         for k=1:feature_number
             data1(j,k)=normrnd(0,1);
         end
end
     