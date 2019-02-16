function CSFN = MeanSquareError(h1,Prices,m)
CSFN= sum((h1-Prices).^2)/(2*m);
end

