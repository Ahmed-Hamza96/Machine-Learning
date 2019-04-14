function [JTheta] = CostFunc(X,Uci)
   [m n] = size(X);
   JTheta = (1/m)*sum((abs(X-Uci)).^2);

end

