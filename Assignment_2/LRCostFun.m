function CSFN = LRCostFun(h,y,m,lamda,Theta)%
CSFN= (1/m)*sum(-y.*log(h)-(1-y).*log(1-h))+(lamda/2*m)*sum(Theta(2:length(Theta)).^2);
end

