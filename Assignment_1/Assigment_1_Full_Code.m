clc
clear all
close all
%Importing Data
%Importing Data
RawData = csvread('house_data_complete.csv',1,2,[1 2 21613 20]);
[R, C]=size(RawData);

%Normalization of all rows
n=C;
for w=1:n
    if max(abs(RawData(:,w)))~=0
    RawData(:,w)=RawData(:,w)./max(RawData(:,w));  
    end
end

%(Training Set) (DS)
DS= floor((60/100)* R);
%(Cross Validation Set) (CVS)
CVS= ceil((20/100)* R);
%(Test Set) (TS)
TS= ceil((20/100)* R);

TrainingSet= RawData(1:DS,:);
CrossVSet= RawData(DS+1:DS+CVS,:);
TestSet=RawData(DS+CVS+1:DS+CVS+TS,:);

%Number of houses
DSHouses= DS;
CVSHouses=CVS;
TSHouses= TS;

%Features of Training Set
DSPrice=TrainingSet(:,1);
DSBedRooms= TrainingSet(:,2);
DSbathrooms= TrainingSet(:,3);
DSSqft_living=TrainingSet(:,4);
DSSqft_Iot=TrainingSet(:,5);
DSFloors= TrainingSet(:,6);
DSWaterFront=TrainingSet(:,7);
DSView=TrainingSet(:,8);
DSCondition=TrainingSet(:,9);
DSGrade=TrainingSet(:,10);
DSSqft_above=TrainingSet(:,11);
DSSqft_Basement=TrainingSet(:,12);
DSYearBuilt=TrainingSet(:,13);
DSyr_renovated=TrainingSet(:,14);
DSzipcode=TrainingSet(:,15);
DSlat=TrainingSet(:,16);
DSLong=TrainingSet(:,17);
DSSqft_living2=TrainingSet(:,18);
DSSqft_Iot15=TrainingSet(:,19);

%Feature
 DSFeatures1 = [DSBedRooms DSbathrooms DSSqft_living DSSqft_above DSSqft_Basement DSYearBuilt DSSqft_living2];
 DSFeatures2 = [DSBedRooms DSbathrooms DSFloors DSSqft_living DSSqft_above DSSqft_Basement DSYearBuilt];
 DSFeatures3 = [DSBedRooms DSbathrooms DSSqft_living DSSqft_above DSSqft_Basement];
 DSFeatures4 = [DSbathrooms DSFloors DSSqft_above DSYearBuilt];
 DSFeatures5 = [DSSqft_living DSSqft_Basement];

%X
 X  = [ones(DS,1) DSFeatures1]; 
 X2 = [ones(DS,1) DSFeatures1 DSFeatures2.^2]; 
 X3 = [ones(DS,1) DSFeatures1.^2 sqrt(DSFeatures2) DSFeatures3]; 
 X4 = [ones(DS,1) DSFeatures3 exp(DSFeatures3) sqrt(DSFeatures2)]; 
 X5 = [ones(DS,1) DSFeatures4 DSFeatures5.^2]; 

n  = length(X(1,:));
n2 = length(X2(1,:));
n3 = length(X3(1,:));
n4 = length(X4(1,:));
n5 = length(X5(1,:));

%Point 1 (Hypothesis Formulation)
    %Thetas
        Theta  = zeros(n,1);
        Theta2 = zeros(n2,1); 
        Theta3 = zeros(n3,1);
        Theta4 = zeros(n4,1);
        Theta5 = zeros(n5,1);
        
    %Hypothesis Function
        h = X * Theta;
        h2= X2 *Theta2;
        h3= X3 *Theta3;
        h4= X4 *Theta4;
        h5= X5 *Theta5;
        
%Point 2 (Mean Square Error)
     Alpha = 0.0001;
     m= DSHouses;
     
    %CostFunction or J(Theta)
    k=1;
    
    E(k)=MeanSquareError(h,DSPrice,m);
    E2(k)=MeanSquareError(h2,DSPrice,m);
    E3(k)=MeanSquareError(h3,DSPrice,m);
    E4(k)=MeanSquareError(h4,DSPrice,m);
    E5(k)=MeanSquareError(h5,DSPrice,m);
    
%Point 3 (Gradient Decent to find the optimal parameters for each hypothesis)
    %1st Hypothesis 
       %Expected Optimal Thetas and Cost Function
            [OptTheta1, Error,itr] = GradientDescent(Theta,Alpha,m,X,DSPrice,E);
            Gradient_Descent_Thetas  =  OptTheta1';
            CostFunc_Of_GD1 = Error(length(Error));
       
             figure(1)
             NoOfIterations=(1:itr);
             plot(NoOfIterations,Error)
             title('First Hypothesis Graph')
             xlabel('No. Of Iterations')
             ylabel('Cost Function J(Theta)')
   
       %Normal Equation of the first hypothesis
            NEq1 = NormalEquation(X,DSPrice);
            hNEq1 =  X * NEq1;
            Normal_Equation_Thetas1 = NEq1' ;
            CostFuncnNeq1 =MeanSquareError(hNEq1,DSPrice,m);
   
   %2nd Hypothesis 
       %Expected Optimal Thetas and Cost Function
            [OptTheta2, Error2, itr2] = GradientDescent(Theta2,Alpha,m,X2,DSPrice,E2);
            Gradient_Descent_Thetas2  =  OptTheta2';
            CostFunc_Of_GD2 = Error2(length(Error2));
  
               figure(2)
               NoOfIterations2=(1:itr2);
               plot(NoOfIterations2,Error2)
               title('Second Hypothesis Graph')
               xlabel('No. Of Iterations')
               ylabel('Cost Function J(Theta)')
               
                %Normal Equation
            NEq2 = NormalEquation(X2,DSPrice);
            hNEq2 =  X2 * NEq2;
            Normal_Equation_Thetas2 = NEq2' ;
            CostFuncnNeq2=MeanSquareError(hNEq2,DSPrice,m);
   
   %3rd Hypothesis
        %Expected Optimal Thetas and Cost Function
             [OptTheta3, Error3, itr3] = GradientDescent(Theta3,Alpha,m,X3,DSPrice,E3);
              Gradient_Descent_Thetas3  =  OptTheta3';
              CostFunc_Of_GD3 = Error3(length(Error3));
   
               figure(3)
               NoOfIterations3=(1:itr3);
               plot(NoOfIterations3,Error3)
               title('Third Hypothesis Graph')
               xlabel('No. Of Iterations')
               ylabel('Cost Function J(Theta)')
               
               %Normal Equation
            NEq3 = NormalEquation(X3,DSPrice);
            hNEq3 =  X3 * NEq3;
            Normal_Equation_Thetas3 = NEq3' ;
            CostFuncnNeq3=MeanSquareError(hNEq3,DSPrice,m);
   
   %4th Hypothesis (Polynomial Regression)
        %Expected Optimal Thetas and Cost Function
            [OptTheta4, Error4, itr4] = GradientDescent(Theta4,Alpha,m,X4,DSPrice,E4);
            Gradient_Descent_Thetas4  =  OptTheta4';
            CostFunc_Of_GD4 = Error4(length(Error4));
   
             figure(4)
             NoOfIterations4=(1:itr4);
             plot(NoOfIterations4,Error4)
             title('Fourth Hypothesis Graph')
             xlabel('No. Of Iterations')
             ylabel('Cost Function J(Theta)')
               
         %Normal Equation
            NEq4 = NormalEquation(X4,DSPrice);
            hNEq4 =  X4 * NEq4;
            Normal_Equation_Thetas4 = NEq4' ;
            CostFuncnNeq4=MeanSquareError(hNEq4,DSPrice,m);
   
    %5th Hypothesis (Polynomial Regression)
        [OptTheta5, Error5, itr5] = GradientDescent(Theta5,Alpha,m,X5,DSPrice,E5);
        Gradient_Descent_Thetas5  =  OptTheta5';
        CostFunc_Of_GD5 = Error5(length(Error5));
   
        figure(5)
        NoOfIterations4=(1:itr5);
        plot(NoOfIterations4,Error5)
        title('Fifth Hypothesis Graph')
        xlabel('No. Of Iterations')
        ylabel('Cost Function J(Theta)')
                 
        %Normal Equation
        NEq5 = NormalEquation(X5,DSPrice);
        hNEq5 =  X5 * NEq5;
        Normal_Equation_Thetas5 = NEq5' ;
        CostFuncnNeq5=MeanSquareError(hNEq5,DSPrice,m);
               
    %Cost function values of all hpothesis
    Iterations = [itr itr2 itr3 itr4 itr5]
    CostFunVectorDSNEq=[CostFuncnNeq1 CostFuncnNeq2 CostFuncnNeq3 CostFuncnNeq4 CostFuncnNeq5]
    CostFuncVectorDSGD=[CostFunc_Of_GD1 CostFunc_Of_GD2 CostFunc_Of_GD3 CostFunc_Of_GD4 CostFunc_Of_GD5]
%--------------------------------------------------------------------------------------------------------------
%Cross Validation Part  
     %Features of Test Set
     CVSPrices=CrossVSet(:,1);
     CVSBedRooms= CrossVSet(:,2);
     CVSbathrooms= CrossVSet(:,3);
     CVSSqft_living=CrossVSet(:,4);
     CVSSqft_Iot=CrossVSet(:,5);
     CVSFloors= CrossVSet(:,6);
     CVSWaterFront=CrossVSet(:,7);
     CVSView=CrossVSet(:,8);
     CVSCondition=CrossVSet(:,9);
     CVSGrade=CrossVSet(:,10);
     CVSSqft_above=CrossVSet(:,11);
     CVSSqft_Basement=CrossVSet(:,12);
     CVSYearBuilt=CrossVSet(:,13);
     CVSyr_renovated=CrossVSet(:,14);
     CVSzipcode=CrossVSet(:,15);
     CVSlat=CrossVSet(:,16);
     CVSLong=CrossVSet(:,17);
     CVSSqft_living2=CrossVSet(:,18);
     CVSSqft_Iot15=CrossVSet(:,19);
    
     %Feature
      CVSFeatures1 = [CVSBedRooms CVSbathrooms CVSSqft_living CVSSqft_above CVSSqft_Basement CVSYearBuilt CVSSqft_living2];
      CVSFeatures2 = [CVSBedRooms CVSbathrooms CVSFloors CVSSqft_living CVSSqft_above CVSSqft_Basement CVSYearBuilt];
      CVSFeatures3 = [CVSBedRooms CVSbathrooms CVSSqft_living CVSSqft_above CVSSqft_Basement];
      CVSFeatures4 = [CVSbathrooms CVSFloors CVSSqft_above CVSYearBuilt];
      CVSFeatures5 = [CVSSqft_living CVSSqft_Basement];
     
     %X
     XCVS  = [ones(CVS,1) CVSFeatures1]; %linear Regression done
     XCVS2 = [ones(CVS,1) CVSFeatures1 CVSFeatures2.^2]; %Polynomial Regression 
     XCVS3 = [ones(CVS,1) CVSFeatures1.^2 sqrt(CVSFeatures2) CVSFeatures3]; 
     XCVS4 = [ones(CVS,1) CVSFeatures3 exp(CVSFeatures3) sqrt(CVSFeatures2)]; 
     XCVS5 = [ones(CVS,1) CVSFeatures4 CVSFeatures5.^2]; 

    %Thetas
      ThetaCVS  = OptTheta1;
      ThetaCVS2 = OptTheta2;
      ThetaCVS3 = OptTheta3;
      ThetaCVS4 = OptTheta4;
      ThetaCVS5 = OptTheta5;
        
    %Hypothesis Function
        hCVS = XCVS * ThetaCVS;
        hCVS2= XCVS2 *ThetaCVS2;
        hCVS3= XCVS3 *ThetaCVS3;
        hCVS4= XCVS4 *ThetaCVS4;
        hCVS5= XCVS5 *ThetaCVS5;
        
%Point 2 (Mean Square Error)
     Alpha = 0.0001;
     mCVS= CVSHouses;
     
    %CostFunction or J(Theta)
    k=1;
    
    %Verification Process of the Hypothesis
        %Cost Function on Test Data
            ECVS(k)=MeanSquareError(hCVS,CVSPrices,mCVS);
            ECVS2(k)=MeanSquareError(hCVS2,CVSPrices,mCVS);
            ECVS3(k)=MeanSquareError(hCVS3,CVSPrices,mCVS);
            ECVS4(k)=MeanSquareError(hCVS4,CVSPrices,mCVS);
            ECVS5(k)=MeanSquareError(hCVS5,CVSPrices,mCVS);
    
    CostFunVectorCVS = [ECVS ECVS2 ECVS3 ECVS4 ECVS5]
%--------------------------------------------------------------------------------------------------------------
%Test Set Part  
  %Features of Test Set
    TSPrices=TestSet(:,1);
    TSBedRooms= TestSet(:,2);
    TSbathrooms= TestSet(:,3);
    TSSqft_living=TestSet(:,4);
    TSSqft_Iot=TestSet(:,5);
    TSFloors= TestSet(:,6);
    TSWaterFront=TestSet(:,7);
    TSView=TestSet(:,8);
    TSCondition=TestSet(:,9);
    TSGrade=TestSet(:,10);
    TSSqft_above=TestSet(:,11);
    TSSqft_Basement=TestSet(:,12);
    TSYearBuilt=TestSet(:,13);
    TSyr_renovated=TestSet(:,14);
    TSzipcode=TestSet(:,15);
    TSlat=TestSet(:,16);
    TSLong=TestSet(:,17);
    TSSqft_living2=TestSet(:,18);
    TSSqft_Iot15=TestSet(:,19);

%Feature
    TSFeatures1 = [TSBedRooms TSbathrooms TSSqft_living TSSqft_above TSSqft_Basement TSYearBuilt TSSqft_living2];
    TSFeatures2 = [TSBedRooms TSbathrooms TSFloors TSSqft_living TSSqft_above TSSqft_Basement TSYearBuilt];
    TSFeatures3 = [TSBedRooms TSbathrooms TSSqft_living TSSqft_above TSSqft_Basement];
    TSFeatures4 = [TSbathrooms TSFloors TSSqft_above TSYearBuilt ];
    TSFeatures5 = [TSSqft_living TSSqft_Basement];
%X
 XTS  = [ones(TS,1) TSFeatures1]; %linear Regression done
 XTS2 = [ones(TS,1) TSFeatures1 TSFeatures2.^2]; %Polynomial Regression 
 XTS3 = [ones(TS,1) TSFeatures1.^2 sqrt(TSFeatures2) TSFeatures3]; 
 XTS4 = [ones(TS,1) TSFeatures3 exp(TSFeatures3) sqrt(TSFeatures2)]; 
 XTS5 = [ones(TS,1) TSFeatures4 TSFeatures5.^2]; 

 %Thetas
        ThetaTS  = OptTheta1;
        ThetaTS2 = OptTheta2;
        ThetaTS3 = OptTheta3;
        ThetaTS4 = OptTheta4;
        ThetaTS5 = OptTheta5;
        
    %Hypothesis Function
        hTS = XTS * ThetaTS;
        hTS2= XTS2 *ThetaTS2;
        hTS3= XTS3 *ThetaTS3;
        hTS4= XTS4 *ThetaTS4;
        hTS5= XTS5 *ThetaTS5;
        
%Point 2 (Mean Square Error)
     Alpha = 0.0001;
     mTS= TSHouses;
     
    %CostFunction or J(Theta)
    k=1;
    
    %Verification Process of the Hypothesis
        %Cost Function on Test Data
            ETS(k)=MeanSquareError(hTS,TSPrices,mTS);
            ETS2(k)=MeanSquareError(hTS2,TSPrices,mTS);
            ETS3(k)=MeanSquareError(hTS3,TSPrices,mTS);
            ETS4(k)=MeanSquareError(hTS4,TSPrices,mTS);
            ETS5(k)=MeanSquareError(hTS5,TSPrices,mTS);
    
    CostFunVectorTS = [ETS ETS2 ETS3 ETS4 ETS5]
%--------------------------------------------------------------------------------------------------------------    
disp('It is clear from the cost function of cross validation vector that the smallest value is')
ErrorCVS = min(CostFunVectorCVS)
disp('which corresponds to the 4th hypothesis. So by applying the Test Set on this hypothesis, the cose function value (error) will be')
ErrorTS = min(CostFunVectorTS)
disp('The difference between the 2 errors is')
Difference=abs(ErrorCVS-ErrorTS)
