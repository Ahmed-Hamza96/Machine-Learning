clc
clear all
close all

%Importing Data
RawData = csvread('heart_DD.csv',1,0,[1 0 250 13]);
[R, C]=size(RawData);

%Normalization of all rows
n=C;
for w=1:n
    if max(abs(RawData(:,w)))~=0 
    RawData(:,w)=RawData(:,w)./max(RawData(:,w));  
    end
end

%(Training Set) (DS)
DS= floor((80/100)* R);
%(Cross Validation Set) (CVS)
CVS= ceil((10/100)* R);
%(Test Set) (TS)
TS= ceil((10/100)* R);

%3 Sets
TrainingSet= RawData(1:DS,:);
CrossVSet= RawData(DS+1:DS+CVS,:);
TestSet=RawData(DS+CVS+1:DS+CVS+TS,:);

%Number of Patients
DSPatients= DS;
CVSPatients=CVS;
TSPatients= TS;
%------------------------------------------------------------------------------------------------------------------------
%Features of Training Set
DSAge=TrainingSet(:,1);
DSSex=TrainingSet(:,2);
DSCp=TrainingSet(:,3);
DSTrestbps=TrainingSet(:,4);
DSChol=TrainingSet(:,5);
DSFbs=TrainingSet(:,6);
DSRestecg=TrainingSet(:,7);
DSThalach=TrainingSet(:,8);
DSExang=TrainingSet(:,9);
DSOldpeak=TrainingSet(:,10);
DSSlope=TrainingSet(:,11);
DSCa=TrainingSet(:,12);
DSthal=TrainingSet(:,13);
DSTarget=TrainingSet(:,14);

%DSFeature
DSFeature1 = [DSAge DSSex DSCp DSTrestbps DSChol DSFbs DSRestecg DSThalach DSExang DSOldpeak DSSlope DSCa DSthal];
DSFeature2 = [DSAge DSSex DSthal DSChol DSFbs DSCa]; 
DSFeature3 = [DSAge DSSex DSCp DSChol DSFbs DSThalach DSOldpeak DSthal]; 
DSFeature4 = [DSAge DSSex DSTrestbps DSChol DSFbs DSExang DSSlope DSCa DSthal];

%X
X  = [ones(DS,1) DSFeature1 exp(DSFeature1) DSFeature1.^4];
X2 = [ones(DS,1) DSFeature2 DSFeature2.^2];
X3 = [ones(DS,1) exp(DSFeature3) DSFeature3.^2];
X4 = [ones(DS,1) DSFeature4.^2 DSFeature4.^4];

n=length(X(1,:));
n2=length(X2(1,:));
n3=length(X3(1,:));
n4=length(X4(1,:));
%Point1 (Hypothesis Formulation)
        %Theta
           Theta  = zeros(n,1);
           Theta2 = zeros(n2,1);
           Theta3 = zeros(n3,1);
           Theta4 = zeros(n4,1);
           
        %Hypothesis Function
           h =  1./(1 + exp(X  * (-Theta)  ));
           h2 = 1./(1 + exp(X2 * (-Theta2) ));
           h3 = 1./(1 + exp(X3 * (-Theta3) ));
           h4 = 1./(1 + exp(X4 * (-Theta4) ));

%Point 2 
    Alpha = 0.003;
    m = DSPatients;
    lamda=0.0001;
    
    %Cost function or J(Theta)
    k=1;
    
    E(k)  = LRCostFun(h ,DSTarget,m,lamda,Theta);
    E2(k) = LRCostFun(h2,DSTarget,m,lamda,Theta2);
    E3(k) = LRCostFun(h3,DSTarget,m,lamda,Theta3);
    E4(k) = LRCostFun(h4,DSTarget,m,lamda,Theta4);
    
    InitialError = [E E2 E3 E4];
    
    %Point 3 (Gradient Decent to find the optimal parameters for each hypothesis)
           %1st Hypothesis (Linear Regression)
               %Expected Optimal Thetas and Cost Function
            [OptTheta1,Error,itr] = GradientDescent2(Theta,Alpha,m,X,DSTarget,E);
            Gradient_Descent_Thetas  =  OptTheta1';
            CostFunc_Of_GD1 = Error(length(Error));
          
            %2nd Hypothesis (Linear Regression)
                %Expected Optimal Thetas and Cost Function
            [OptTheta2,Error2,itr2] = GradientDescent2(Theta2,Alpha,m,X2,DSTarget,E2);
            Gradient_Descent_Thetas2 =  OptTheta2';
            CostFunc_Of_GD2 = Error2(length(Error2));
    
             %3rd Hypothesis (Linear Regression)
                %Expected Optimal Thetas and Cost Function
            [OptTheta3, Error3,itr3] = GradientDescent2(Theta3,Alpha,m,X3,DSTarget,E3);
            Gradient_Descent_Thetas3  =  OptTheta3';
            CostFunc_Of_GD3 = Error3(length(Error3));
          
            %4th Hypothesis (Linear Regression)
                %Expected Optimal Thetas and Cost Function
            [OptTheta4, Error4,itr4] = GradientDescent2(Theta4,Alpha,m,X4,DSTarget,E4);
            Gradient_Descent_Thetas4  =  OptTheta4';
            CostFunc_Of_GD4 = Error4(length(Error4));
           
             %No Of Iterations
                NoOfIterations =(1:itr);
                NoOfIterations2=(1:itr2);
                NoOfIterations3=(1:itr3);
                NoOfIterations4=(1:itr4);
  
              %Plotting of No. of Iterations of each hypothesis with its
              %error
                figure(1)
                subplot (2,2,1),plot(NoOfIterations,Error),
                title('First Hypothesis Graph'),
                xlabel('No. Of Iterations'),
                ylabel('Cost Function J(Theta)'),
                
                subplot (2,2,2),plot(NoOfIterations2,Error2)
                title('Second Hypothesis Graph')
                xlabel('No. Of Iterations')
                ylabel('Cost Function J(Theta)'),
                
                subplot (2,2,3)
                plot(NoOfIterations3,Error3)
                title('Third Hypothesis Graph')
                xlabel('No. Of Iterations')
                ylabel('Cost Function J(Theta)')
                
                subplot(2,2,4),
                plot(NoOfIterations4,Error4)
                title('Fourth Hypothesis Graph'),
                xlabel('No. Of Iterations'),
                ylabel('Cost Function J(Theta)')
             
                ErrorGD = [CostFunc_Of_GD1 CostFunc_Of_GD2 CostFunc_Of_GD3 CostFunc_Of_GD4]
%------------------------------------------------------------------------------------------------------------------------
%Cross Validation Part               
 CVSAge=CrossVSet(:,1);
 CVSSex=CrossVSet(:,2);
 CVSCp=CrossVSet(:,3);
 CVSTrestbps=CrossVSet(:,4);
 CVSChol=CrossVSet(:,5);
 CVSFbs=CrossVSet(:,6);
 CVSRestecg=CrossVSet(:,7);
 CVSThalach=CrossVSet(:,8);
 CVSExang=CrossVSet(:,9);
 CVSOldpeak=CrossVSet(:,10);
 CVSSlope=CrossVSet(:,11);
 CVSCa=CrossVSet(:,12);
 CVSthal=CrossVSet(:,13);
 CVSTarget=CrossVSet(:,14);    

%CVSFeature
CVSFeature1 = [CVSAge CVSSex CVSCp CVSTrestbps CVSChol CVSFbs CVSRestecg CVSThalach CVSExang CVSOldpeak CVSSlope CVSCa CVSthal];
CVSFeature2 = [CVSAge CVSSex  CVSthal CVSChol CVSFbs CVSCa];
CVSFeature3 = [CVSAge CVSSex  CVSCp CVSChol CVSFbs CVSThalach CVSOldpeak CVSthal]; %DSCp DSTrestbps
CVSFeature4 = [CVSAge CVSSex  CVSTrestbps CVSChol CVSFbs CVSExang CVSSlope CVSCa CVSthal];

%Xcvs
XCVS  = [ones(CVS,1) CVSFeature1 exp(CVSFeature1) CVSFeature1.^4 ];
XCVS2 = [ones(CVS,1) CVSFeature2 CVSFeature2.^3];
XCVS3 = [ones(CVS,1) CVSFeature3 CVSFeature3.^2];
XCVS4 = [ones(CVS,1) CVSFeature4.^2 CVSFeature4.^4];

nCVS =length(XCVS(1,:));
nCVS2=length(XCVS2(1,:));
nCVS3=length(XCVS3(1,:));
nCVS4=length(XCVS4(1,:));
%Point1 (Hypothesis Formulation)
    %Theta
        ThetaCVS   = OptTheta1;
        ThetaCVS2  = OptTheta2;
        ThetaCVS3  = OptTheta3;
        ThetaCVS4  = OptTheta4;
    %Hypothesis Function
           hCVS  = 1./(1 + exp(XCVS  * (-ThetaCVS)  ));
           hCVS2 = 1./(1 + exp(XCVS2 * (-ThetaCVS2) ));
           hCVS3 = 1./(1 + exp(XCVS3 * (-ThetaCVS3) ));
           hCVS4 = 1./(1 + exp(XCVS4 * (-ThetaCVS4) ));
         
%Point 2 
    mCVS = CVSPatients;
    %Cost function or J(Theta)
    k=1;
    
    ECVS(k) = LRCostFun(hCVS,CVSTarget,mCVS,lamda,ThetaCVS);
    ECVS2(k) = LRCostFun(hCVS2,CVSTarget,mCVS,lamda,ThetaCVS2);
    ECVS3(k) = LRCostFun(hCVS3,CVSTarget,mCVS,lamda,ThetaCVS3);
    ECVS4(k) = LRCostFun(hCVS4,CVSTarget,mCVS,lamda,ThetaCVS4);
    
   ErrorCVS=[ECVS ECVS2 ECVS3 ECVS4] 
%------------------------------------------------------------------------------------------------------------------------   
%Test Part
TSAge=TestSet(:,1);
TSSex=TestSet(:,2);
TSCp=TestSet(:,3);
TSTrestbps=TestSet(:,4);
TSChol=TestSet(:,5);
TSFbs=TestSet(:,6);
TSRestecg=TestSet(:,7);
TSThalach=TestSet(:,8);
TSExang=TestSet(:,9);
TSOldpeak=TestSet(:,10);
TSSlope=TestSet(:,11);
TSCa=TestSet(:,12);
TSthal=TestSet(:,13);
TSTarget=TestSet(:,14);    

%TSFeature
TSFeature1 = [TSAge TSSex TSCp TSTrestbps  TSChol TSFbs TSRestecg TSThalach TSExang TSOldpeak TSSlope TSCa TSthal];
TSFeature2 = [TSAge TSSex  TSthal TSChol TSFbs TSCa];
TSFeature3 = [TSAge TSSex  TSCp TSChol TSFbs TSThalach TSOldpeak TSthal]; %DSCp DSTrestbps
TSFeature4 = [TSAge TSSex  TSTrestbps TSChol TSFbs TSExang TSSlope TSCa TSthal];

%X
XTS  = [ones(TS,1) TSFeature1  exp(TSFeature1) TSFeature1.^4];
XTS2 = [ones(TS,1) TSFeature2 TSFeature2.^3];
XTS3 = [ones(TS,1) TSFeature3 TSFeature3.^2];
XTS4 = [ones(TS,1) TSFeature4.^2 TSFeature4.^4];

nTS =length(XTS(1,:));
nTS2=length(XTS2(1,:));
nTS3=length(XTS3(1,:));
nTS4=length(XTS4(1,:));

%Point1 (Hypothesis Formulation)
        %Theta
        ThetaTS  = OptTheta1;
        ThetaTS2  = OptTheta2;
        ThetaTS3  = OptTheta3;
        ThetaTS4  = OptTheta4;
        %Hypothesis Function
           hTS  = 1./(1 + exp(XTS  * (-ThetaTS)));
           hTS2 = 1./(1 + exp(XTS2 * (-ThetaTS2)));
           hTS3 = 1./(1 + exp(XTS3 * (-ThetaTS3)));
           hTS4 = 1./(1 + exp(XTS4 * (-ThetaTS4)));
%Point 2 (Mean Square Error)    
    mTS = TSPatients;
    %Cost function or J(Theta)
    k=1;
    
    ETS(k) = LRCostFun(hTS,TSTarget,mTS,lamda,ThetaTS);
    ETS2(k) = LRCostFun(hTS2,TSTarget,mTS,lamda,ThetaTS2);
    ETS3(k) = LRCostFun(hTS3,TSTarget,mTS,lamda,ThetaTS3);
    ETS4(k) = LRCostFun(hTS4,TSTarget,mTS,lamda,ThetaTS4);

    ErrorTS= [ETS ETS2 ETS3 ETS4]
    %Cost function value of DS,CVS & TS with each hypothesis
    figure(2)
    plot([1 2 3 4],ErrorGD,'x')
    hold on
    plot([1 2 3 4],ErrorCVS,'o')
    hold on
    plot([1 2 3 4],ErrorTS,'.')
    legend('GD Error','cvs Error','TS Error')
    title('Hypothesis vs Cost Function Values')
    xlabel('Hypothesis')
    ylabel('Cost Function Values')
%------------------------------------------------------------------------------------------------------------------------       
    disp('It is clear that from these results that the least error can be acheived from the first hypothesis which is equal to')
    MinErrorGD= min(ErrorGD)
    disp('And to confirm that the best hypothesis is the first one i used 10% of data in the cross validation part as well as 10% for testing part, The cost function value from the cross validation set and cost function value from the test set ')
    MinErrorCVS=min(ErrorCVS)
    MinErrorTS=min(ErrorTS)