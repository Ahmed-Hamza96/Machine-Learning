clc
clear all
close all

%Importing Data (X) from column 4 to column 21
x = csvread('house_prices_data_training_data.csv',1,2,[1 2 17999 20]);

x_input = x(:,2:19);
[m1 n1]=size(x_input);

%Normalizing the Inputs
Normalized_X = Normalize(x_input,n1);
Normalized_Y = Normalize(x(:,1),1);

% Correlation Matrix of x
Corr_x = corr(Normalized_X);

% Covariance Matrix
x_Conv = cov(Normalized_X);

%Principle Component Analysis
% SVD(X) produces a diagonal matrix S, of the same 
%   dimension as X and with nonnegative diagonal elements in
%   decreasing order, and unitary matrices U and V so that (U*U' = I)
%   X = U*S*V'.
[U, S , V] = svd(x_Conv);

%Eigen Values are the diagonal of Matrix S
EigenValues = diag(S);

%Find K that would make alpha < 0.001

alpha = 1;
k=1;
[m n]=size(S);

while(true)
    Nominator = sum(EigenValues(1:k));
    Denominator = sum(EigenValues(1:m));
    alpha = 1 - (Nominator / Denominator);
    
    if(alpha <= 0.001)
        break;
    end
    
    k=k+1;
end

disp('The value of k is')
 k

% Reduced Dataset
 Reduced_Data = U(:,1:k)'*Normalized_X';
%  size(Reduced_Data);
 
% Approximate Data
% disp('Trans(Reduced Data)');
% size((Reduced_Data)');
% disp('Trans(V(:,1:k))');
% size(V(:,1:k)');
App_Data = Reduced_Data' * V(:,1:k)';

% disp('Approximated Data Set');
% size(App_Data);

%Error
% disp('Trans(App_Data(:,1:k))');
% size(App_Data(:,1:k)');
% disp('Reduced_Data');
% size(Reduced_Data);
% disp('Error')

error = (1/m1)*sum(App_Data(:,1:k)'-Reduced_Data);
Reduced_Data = Reduced_Data';

fprintf('Program paused. Press enter to continue to the Linear Regression.\n');
disp('-----------------------------------------------------------------------');
pause;

%(==========================================LinearRegression=====================================)
%Linear Regression on Reduced_Data
    %Number of houses
        Houses = length(Reduced_Data);
    %Features of Training Set
        Feature1 = Reduced_Data;     
     
     %X
        X=[ones(Houses,1),Feature1];
        nZeros  = length(X(1,:));
        
     %(Hypothesis Formulation)
         %Thetas   
            Theta  = zeros(nZeros,1);
        
         %Hypothesis Function
            h = X * Theta;
        
    %(Mean Square Error)
     Alpha = 0.001;
     m = Houses;
     
    %CostFunction or J(Theta)
    k_new=1;
    Prices = Normalized_Y;
    
    disp('Intial CostFunc. value.')
    E(k_new)=MeanSquareError(h,Prices,Houses)
    
    %(Gradient Decent to find the optimal parameters for each hypothesis)
    %1st Hypothesis 
       %Expected Optimal Thetas and Cost Function
            [OptTheta1, Error,itr] = GradientDescent(Theta,Alpha,Houses,X,Prices,E);
            Gradient_Descent_Thetas  =  OptTheta1';
            CostFunc_Of_GD1 = Error(length(Error));

fprintf('Program paused. Press enter to continue to the K-Means Clustering on reduced data.\n');
disp('-----------------------------------------------------------------------------------');
pause;
%(=======================================K-Means Clustering========================================)
  disp('Loading K-means on reduced data..................................................')
    %K_Means Clustering on Reduced Data
       Reduced_Data;
       ClusterCosts = [];
       JTheta = [];
       %Assuming that i have x clusters
            NoOfClusters = 2;
           
            
   for q = 1:10
       
      for i =1 :100
        [Cluster,U,Uci]=K_means(NoOfClusters,Reduced_Data);
        JTheta = [JTheta CostFunc(Reduced_Data,Uci)];  
      end
      
      LowestCost = min(JTheta);

      %ClusterCosts That gives the lowest cost 
      ClusterCosts = [ClusterCosts LowestCost];
   end   
   
   ClusterCosts
   NoOfIterations = (1:1:10);

   figure(2)
      plot(NoOfIterations,smooth(ClusterCosts))
      title('No of Clusters Vs. Cost Function J(Theta) Reduced Data')
      xlabel('No of Clusters')
      ylabel('Cost Function J(Theta)')
      disp('Respective Center of U in reduced data')
      U
%(------------------------------------------------------------------------------------------) 
fprintf('Program paused. Press enter to continue to the K-Means Clustering on normalized data.\n');
disp('------------------------------------------------------------------------------------------');
pause;
disp('Loading K-means on normalized data........................................................')  
%K_Means Clustering on Normalized Data
             Normalized_X;
             JTheta1 = [];
       ClusterCosts1 = [];
       %Assuming that i have x clusters
            NoOfClusters1 = 2;
            
   for q = 1:10
       
      for i =1 :100
        [Cluster1,U1,Uci1]=K_means(NoOfClusters1,Normalized_X);
        JTheta1 =[JTheta1 CostFunc(Normalized_X,Uci1)];  
      end
      
      LowestCost1 = min(JTheta1);

      %ClusterCosts That gives the lowest cost 
      ClusterCosts1 = [ClusterCosts1 LowestCost1];
   end   
   
   ClusterCosts1
   NoOfIterations1 = (1:1:10);

   figure(3)
      plot(NoOfIterations1,smooth(ClusterCosts))
      title('No of Clusters Vs. Cost Function J(Theta) Normalized Data')
      xlabel('No of Clusters')
      ylabel('Cost Function J(Theta)')
      disp('Respective Center of U in Normalized Data')
      U1
      
fprintf('Program paused. Press enter to continue to the Anomaly detection.\n');
disp('-----------------------------------------------------------------------');
pause;

%(=========================================Anomaly detection=====================================)    
  %Loading the Data
    x_input;
    TS = x_input;
    [m n] = size(TS);

  %Calculating mu
    MEAN = mean(TS);

  %Calculating Sigma
    SIGMA = var(TS);

  %Calculating Covariance
    Covariance = cov(x_input);
    
  %Assume 
    value = 10^-25;
    F = 0;

    P =[];
 %Calculate P(X)
    for i = 1 : m
        INPUTVec = TS(i,:);
        %Denominator1 = power((2*pi),(n/2))*(det(Covariance)^0.5);
        %Nominator2=(-0.5)*(INPUTVec-MEAN)'*(INPUTVec-MEAN)*(inv(Covariance));
        %F = (1./Denominator1)*exp(Nominator2);

        F=mvnpdf(INPUTVec,MEAN,Covariance);
        P = [P F];

        if(F > value)
            %Not Anomaly = 1
            Result(i) = 1;
        else
            %Anomaly = 0
            Result(i) = 0;
        end
    end

disp('Result of the anomaly detection system')
Result 

AnomalySum=sum(Result)

