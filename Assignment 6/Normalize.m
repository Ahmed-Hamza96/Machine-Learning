function X = Normalize(X,n)
for w=1:n
    if max(abs(X(:,w)))~=0
    %X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));  
    X(:,w)=X(:,w)./max(X(:,w));    
    end
end
end
