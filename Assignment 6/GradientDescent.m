function [Theta, E, c] = GradientDescent(Theta,Alpha,m,NormX,NormalizedPrices,E)
c=1;
b=1;
h = NormX * Theta;
  
while b == 1
        Theta = Theta -(Alpha/m)*NormX'*(h-NormalizedPrices);
        c=c+1;
       
        h = NormX * Theta;
        E(c) = MeanSquareError(h,NormalizedPrices,m);
        
        %if the new error > old error stop
            if E(c-1)-E(c)<0
                disp('Exit because of the 1st condition');
                break;
            end 

        %If old error - new error < a certain value stop
            q=(E(c-1)-E(c))./E(c-1);
            if q <.0001;
                b=0;
                disp('Exit because of the 2nd condition');
            end
end
        
      figure(1)
      NoOfIterations=(1:c);
    plot(NoOfIterations,E)
    title('No. Of Iterations Vs Cost Function')
    xlabel('No. Of Iterations')
    ylabel('Cost Function J(Theta)')
   


end

