function [hatbeta_T,hatbeta_S1,hatbeta_S2]=Cp_transfer_solution_multisource_new(X1,Y1,X2,Y2,X3,Y3,lambda_1,lambda_2,lambda_3,lambda_4,lambda_5,feature_number)

for i=1:feature_number
    clear A;
    clear b;
    clear solution;
    A=[lambda_1*X1(:,i)'*X1(:,i)+lambda_2*X2(:,i)'*X2(:,i)+lambda_3*X3(:,i)'*X3(:,i) lambda_1*X1(:,i)'*X1(:,i) lambda_2*X2(:,i)'*X2(:,i) lambda_3*X3(:,i)'*X3(:,i);2*lambda_1*X1(:,i)'*X1(:,i) 2*lambda_1*X1(:,i)'*X1(:,i)+lambda_4(i) 0 0;2*lambda_2*X2(:,i)'*X2(:,i) 0 2*lambda_2*X2(:,i)'*X2(:,i)+lambda_4(i) 0;2*lambda_3*X3(:,i)'*X3(:,i) 0 0 2*lambda_3*X3(:,i)'*X3(:,i)+lambda_4(i)];
    b=[lambda_1*Y1'*X1(:,i)+lambda_2*Y2'*X2(:,i)+lambda_3*Y3'*X3(:,i) 2*lambda_1*Y1'*X1(:,i) 2*lambda_2*Y2'*X2(:,i) 2*lambda_3*Y3'*X3(:,i)]';
    
    B=[A,b];
    
    solution=A\b;
    alpha_1=solution(1)+solution(2);
    alpha_2=solution(1)+solution(3);
    alpha_3=solution(1)+solution(4);
    v_1=solution(2);
    v_2=solution(3);
    v_3=solution(4);
    
    if lambda_1*[norm(Y1-X1(:,i)*alpha_1)]^2+lambda_2*[norm(Y2-X2(:,i)*alpha_2)]^2+lambda_3*[norm(Y3-X3(:,i)*alpha_3)]^2+0.5*lambda_4(i)*(v_1^2+v_2^2+v_3^2)+lambda_5<lambda_1*Y1'*Y1+lambda_2*Y2'*Y2+lambda_3*Y3'*Y3
       hatbeta_T(i)=alpha_1;
       hatbeta_S1(i)=alpha_2;
       hatbeta_S2(i)=alpha_3;
       else
        hatbeta_T(i)=0;
        hatbeta_S1(i)=0;
        hatbeta_S2(i)=0;
    end
    
end