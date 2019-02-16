function [Theta, E, k] = GradientDescent2(Theta,Alpha,m,X,y,E)
k=1;
b=1;
lamda=0.0001;
h = 1./(1 + exp(X*-Theta));
  
while b == 1
        Theta= Theta*(1 - ((Alpha*lamda)/m)) -(Alpha/m)*X'*(h-y);
        k=k+1;
       
        h = 1./(1 + exp(X*(-Theta)));
        E(k)=LRCostFun(h,y,m,lamda,Theta);%,lamda
        
        %if the new error > old error stop
            if E(k-1)-E(k)<0
                break
                'Exit because of the 1st condition';
            end 

        %If old error - new error < a certain value stop
            q=(E(k-1)-E(k))./E(k-1);
            if q <.000001;
                b=0;
                'Exit because of the 2nd condition';
            end
end
end