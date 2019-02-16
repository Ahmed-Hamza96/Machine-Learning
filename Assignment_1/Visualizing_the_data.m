
close all
clc
%Importing Data
 RawData = csvread('house_data_complete.csv',1,2,[1 2 21613 20]);

% y^(i)
Prices=RawData(:,1); 
%Number of houses
Houses= length(RawData(:,1)); 

%Features
BedRooms= RawData(:,2);
bathrooms= RawData(:,3);
Sqft_living=RawData(:,4);
Sqft_Iot=RawData(:,5);
Floors= RawData(:,6);
WaterFront=RawData(:,7);
View=RawData(:,8);
Condition=RawData(:,9);
Grade=RawData(:,10);
Sqft_above=RawData(:,11);
Sqft_Basement=RawData(:,12);
YearBuilt=RawData(:,13);
yr_renovated=RawData(:,14);
zipcode=RawData(:,15);
lat=RawData(:,16);
Long=RawData(:,17);
Sqft_living2=RawData(:,18);
Sqft_Iot15=RawData(:,19);

%Plotting Prices Vs  Values
figure(1)
plot(BedRooms,Prices,'.')
% hold on
% plot(bathrooms,Prices,'x')
% hold on
% plot(Sqft_living,Prices,'o')
% hold on
% plot(Sqft_above,Prices,'o')
% hold on
% plot(Sqft_Basement,Prices,'o')
% hold on
% plot(WaterFront,Prices,'o')
 xlabel('Bedrooms') 
 ylabel('Prices')

figure(2)
plot(bathrooms,Prices,'.')
ylabel('Prices')
xlabel('Bathrooms') 

figure(3)
plot(Sqft_living,Prices,'.')
ylabel('Prices')
xlabel('Sqft living') 

figure(4)
plot(Sqft_Iot,Prices,'.')
ylabel('Prices')
xlabel('Sqft Iot') 


figure(5)
plot(Floors,Prices,'.')
ylabel('Prices')
xlabel('Floors') 


figure(6)
plot(WaterFront,Prices,'.')
ylabel('Prices')
xlabel('WaterFront') 


figure(7)
plot(View,Prices,'.')
ylabel('Prices')
xlabel('View') 


figure(8)
plot(Condition,Prices,'.')
ylabel('Prices')
xlabel('Condition') 


figure(9)
plot(Grade,Prices,'.')
ylabel('Prices')
xlabel('Grade') 


figure(10)
plot(Sqft_above,Prices,'.')
ylabel('Prices')
xlabel('Sqft above') 


figure(11)
plot(Sqft_Basement,Prices,'.')
ylabel('Prices')
xlabel('Sqft Basement') 


figure(12)
plot(YearBuilt,Prices,'.')
ylabel('Prices')
xlabel('YearBuilt') 


figure(13)
plot(yr_renovated,Prices,'.')
ylabel('Prices')
xlabel('yr renovated') 


figure(14)
plot(zipcode,Prices,'.')
ylabel('Prices')
xlabel('zipcode') 


figure(15)
plot(lat,Prices,'.')
ylabel('Prices')
xlabel('lat') 


figure(16)
plot(Long,Prices,'.')
ylabel('Prices')
xlabel('Long') 


figure(17)
plot(Sqft_living2,Prices,'.')
ylabel('Prices')
xlabel('Sqft living2') 

figure(18)
plot(Sqft_Iot15,Prices,'.')
ylabel('Prices')
xlabel('Sqft Iot15') 


