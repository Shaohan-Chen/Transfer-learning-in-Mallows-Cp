function [Y,tildeY]=build_data2(num_train_1,data1,beta,delta)

        for l=1:num_train_1
            e(l)=normrnd(0,1);
        end
        
        for l=1:num_train_1
            eta(l)=normrnd(0,1);
        end
        
        Y=data1*(beta)+e';
        tildeY=data1*(beta+delta)+eta';