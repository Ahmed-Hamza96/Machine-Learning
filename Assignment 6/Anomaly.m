clear all
close all
clc

x = csvread('house_prices_data_training_data.csv',1,2,[1 2 17999 20]);
x_input = x(:,2:19);
[m1 n1]=size(x_input);       

%(=========================================Anomaly detection=====================================)    
  %Loading the Data
    x_input;
    TS = x_input;
    [m n] = size(TS);

  %Calculating U
    MEAN = mean(TS);

  %Calculating Sigma
    SIGMA = var(TS);

  %Calculating Covariance
    Covariance = cov(x_input);

    j = 15;
    v=x_input(15,:);
    
    %Calculate PDF
     CDF = normcdf(v,MEAN,SIGMA);

    %Assume
    value = 0.001;
    
    Prod_CDF = prod(CDF)
   
    AnomalyDetect =  Prod_CDF >= value || Prod_CDF < value   
    