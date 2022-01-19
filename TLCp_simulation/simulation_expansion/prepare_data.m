function X=prepare_data(feature_number,trial_number)

for i=1:trial_number
   for j=1:feature_number
      Y(i,j)=normrnd(0,1);
   end
end

[V,D]=eig(Y'*Y);
Q=V*D^(-1/2)*sqrt(trial_number);

X=Y*Q;