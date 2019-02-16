function NE = NormalEquation(X,Y)

NE = (inv((X'* X)))*X'*Y;

end

