clc
clear all
close all
%Importing Data
RawData = csvread('heart_DD.csv',1,0,[1 0 250 13]);
[R, C]=size(RawData);

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
DSHouses= DS;
CVSHouses=CVS;
TSHouses= TS;

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

figure(1)
plot(DSAge,DSTarget,'o')
figure(2)
plot(DSSex,DSTarget,'o')
figure(3)
plot(DSCp,DSTarget,'o')
figure(4)
plot(DSTrestbps,DSTarget,'o')
figure(5)
plot(DSTrestbps,DSTarget,'o')
figure(6)
plot(DSChol,DSTarget,'o')
figure(7)
plot(DSFbs,DSTarget,'o')
figure(8)
plot(DSRestecg,DSTarget,'o')
figure(9)
plot(DSThalach,DSTarget,'o')
figure(10)
plot(DSExang,DSTarget,'o')
figure(11)
plot(DSOldpeak,DSTarget,'o')
figure(12)
plot(DSSlope,DSTarget,'o')
figure(13)
plot(DSCa,DSTarget,'o')
figure(14)
plot(DSthal,DSTarget,'o')

 