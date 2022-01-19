clear;
num_train_1=30;
num_train_2=30;
feature_number=300;

beta=zeros(feature_number,1);

beta(2)=-0.1245;
beta(6)=0.4552;
beta(12)=0.9619;
beta(20)=0.3251;

beta0=zeros(feature_number,1);
beta0(2)=0.1245;
beta0(8)=0.4552;
beta0(12)=0.9619;
beta0(20)=-0.3251;

delta0=beta0;

[data1]=build_data1(num_train_1,feature_number);
[V1,D1] = eig(data1'*data1);

%find out the non-zero eighenvalue and eigenvector
j=1;
for i=1:size(data1'*data1,2)
    if abs(D1(i,i))>0.0001
       V(:,j)=V1(:,i)*D1(i,i)^(-1/2);
       j=j+1;
    end
end
Q_hat1=V*sqrt(num_train_1);
data_schmidt1=data1*Q_hat1;
%load('simulation_high_dimensional_test');
tic
index=1;
for d=1:6
    mse_Cp=0;
    mse_Cp_cutoff=0;
    mse_TLCp=0;
    mse_TLCp_cutoff=0;
    mse_least_square=0;
    parfor try_number=1:5000
        [mse_least]=simulation_least(num_train_1,data1,beta,delta0);
        [mse2,mse2_cutoff]=simulation_approximate_Cp(num_train_1,feature_number,data1,beta,Q_hat1,data_schmidt1);
        [mse4,mse4_cutoff]=simulation_approximate_TLCp(num_train_1,feature_number,data1,beta,delta0,d,Q_hat1,data_schmidt1);
        mse_Cp=mse_Cp+mse2;
        mse_Cp_cutoff=mse_Cp_cutoff+mse2_cutoff;
        mse_TLCp=mse_TLCp+mse4;
        mse_TLCp_cutoff=mse_TLCp_cutoff+mse4_cutoff;
        mse_least_square=mse_least_square+mse_least;
    end
    try_number=5000;
    mse_app_Cp(index)=mse_Cp/try_number;
    mse_app_Cp_cutoff(index)=mse_Cp_cutoff/try_number;
    mse_app_TLCp(index)=mse_TLCp/try_number;
    mse_app_TLCp_cutoff(index)=mse_TLCp_cutoff/try_number;
    mse_least(index)=mse_least_square/try_number;
    index=index+1;
    d
end
toc
save simulation_high_dimensional mse_app_Cp mse_app_Cp_cutoff mse_app_TLCp mse_app_TLCp_cutoff
