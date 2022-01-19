function [tildeY]=preparedata22(tildedata_size,delta,tildeX,beta)

for i=1:tildedata_size
    eta(i)=normrnd(0,1);
end

tildeY=tildeX*(beta+delta)+eta';