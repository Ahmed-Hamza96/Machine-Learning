function [Theta, E, k] = GradientDescent(Theta,Alpha,m,NormX,NormalizedPrices,E)
k=1;
b=1;
  h = NormX * Theta;
  
while b == 1
        Theta= Theta -(Alpha/m)*NormX'*(NormX*Theta-NormalizedPrices);
        k=k+1;
       
        h = NormX * Theta;
        E(k)=MeanSquareError(h,NormalizedPrices,m);
        
        %if the new error > old error stop
            if E(k-1)-E(k)<0
                break
                'Exit because of the 1st condition';
            end 

        %If old error - new error < a certain value stop
            q=(E(k-1)-E(k))./E(k-1);
            if q <.00001;
                b=0;
                'Exit because of the 2nd condition';
            end
end

end

