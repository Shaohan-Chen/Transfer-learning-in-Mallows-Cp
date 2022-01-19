function binary_representation=convert_to_binary(count,k)
%k,the dimension of the regression model;
    b=dec2bin(count); % convert to its binary representation;
    c=size(b);
    d=c(2);% the length of the b;
    for i=1:d
    number(i)=str2num(b(i));
    end
    
    % adding zeros in the front of the b;
    if d<k
        
          for i=1:d
              binary_representation(i+k-d)=number(i);
          end
          for i=1:k-d
              binary_representation(i)=0;  
          end
    else   
          binary_representation=number;   
    end
    