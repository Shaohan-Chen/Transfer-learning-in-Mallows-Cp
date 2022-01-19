function [Y]=preparedata21(data_size,X,beta)

for i=1:data_size
    e(i)=normrnd(0,1);
end

Y=X*beta+e';
