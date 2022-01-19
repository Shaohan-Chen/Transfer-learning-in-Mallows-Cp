function [mse_least]=simulation_least(num_train_1,data1,beta,delta0)

[Y1,Y2]=build_data2(num_train_1,data1,beta,delta0);

tildebeta=data1'*(data1*data1')^(-1)*Y1;
mse_least=(tildebeta-beta)'*(tildebeta-beta);