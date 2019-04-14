function [Cluster,U,Uci] = K_means(NoOfClusters,Reduced_Data)
[R C] = size(Reduced_Data);
Limit = R;
U = zeros(C,NoOfClusters);
  
    %Defining No. of clusters
     for t = 1 : NoOfClusters
                U_new = Reduced_Data(randi([1,Limit],1,1),:);
                U(:,t) = U_new';
     end
          
         Cluster = zeros(Limit,1);
  
            for i = 1 : Limit
                
       %Cluster Assignment
                for j = 1 : NoOfClusters
                
                    Result = sum(abs((Reduced_Data(i,:)-U(:,j)').^2));
                    MinVec(j) = Result ;
                    
                end
                MinDist = find(MinVec==min(MinVec));
                Cluster(i) = MinDist;
        
            end
       ClusterRow = Cluster';
       
       %Centroid Adjustement
      for w = 1 : NoOfClusters 
  
       IndC = find(ClusterRow == w);
       ValuesRed_Data = Reduced_Data(IndC,:);
       U(:,w) = mean(ValuesRed_Data) ;
       
      end
      
      %cluster centroid of cluster to which the example x^((i)) has been assigned

      Uci = [];
      
      for q = 1 : Limit
        Uci(:,q)=U(:,ClusterRow(q));
      end
      
      Uci = Uci';
end

