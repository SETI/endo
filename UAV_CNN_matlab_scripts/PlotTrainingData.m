
%% for 3cm network
load('RGBNet3cmE1to30.mat');
figure;
subplot(2,1,1);
plot(RGBNet3cmE1to30Info.iterations,...
    RGBNet3cmE1to30Info.TrainingAccuracy);hold on;
plot(RGBNet3cmE1to30Info.iterations,...
    smooth(RGBNet3cmE1to30Info.TrainingAccuracy));hold on;
plot(RGBNet3cmE1to30Info.valIterations,...
    RGBNet3cmE1to30Info.ValidationAccuracy,...
    'k--','LineWidth',1);
subplot(2,1,2);
plot(RGBNet3cmE1to30Info.iterations,...
    RGBNet3cmE1to30Info.TrainingLoss);hold on;
plot(RGBNet3cmE1to30Info.iterations,...
    smooth(RGBNet3cmE1to30Info.TrainingLoss));hold on;
plot(RGBNet3cmE1to30Info.valIterations,...
    RGBNet3cmE1to30Info.ValidationLoss,...
    'k--','LineWidth',1);

%% for normal stuff that you did't have to patch together
% load('RGBNet30p9cmE1to50.mat');
info = RGBNetwork17p1cmE1to100Info;

tAcc = info.TrainingAccuracy;
tLoss = info.TrainingLoss;
vAcc = info.ValidationAccuracy;
vLoss = info.ValidationLoss;
xVal = 1:length(vAcc);
xVal(isnan(vAcc)) = [];
vAcc(isnan(vAcc)) = [];
vLoss(isnan(vLoss)) = [];
figure;

subplot(2,1,1);

% plot(RGBNet3cmE1to30Info.iterations,...
%     RGBNet3cmE1to30Info.TrainingAccuracy);hold on;
% plot(RGBNet3cmE1to30Info.iterations,...
%     smooth(RGBNet3cmE1to30Info.TrainingAccuracy));hold on;
% plot(RGBNet3cmE1to30Info.valIterations,...
%     RGBNet3cmE1to30Info.ValidationAccuracy,...
%     'k--','LineWidth',1);
% s=ceil(RGBNet3cmE1to30Info.iterations(end)/length(tAcc));
plot(1:s:s*length(tAcc),tAcc);hold on;
plot(1:s:s*length(tAcc),smooth(tAcc));hold on;
plot(1:s*(xVal(4)-xVal(3)):s*xVal(end)+1, vAcc,'k--','LineWidth',2)

subplot(2,1,2);
% 
% plot(RGBNet3cmE1to30Info.iterations,...
%     RGBNet3cmE1to30Info.TrainingLoss);hold on;
% plot(RGBNet3cmE1to30Info.iterations,...
%     smooth(RGBNet3cmE1to30Info.TrainingLoss));hold on;
% plot(RGBNet3cmE1to30Info.valIterations,...
%     RGBNet3cmE1to30Info.ValidationLoss,...
%     'k--','LineWidth',1);

plot(1:s:s*length(tLoss),tLoss);hold on;
plot(1:s:s*length(tLoss),smooth(tLoss)); hold on;
plot(1:s*(xVal(4)-xVal(3)):s*xVal(end)+1, vLoss,'k--', 'LineWidth', 2)




%% for normal stuff that you did't have to patch together
% load('RGBNet30p9cmE1to50.mat');
info = RGBNetwork17p1cmE1to100Info;

tAcc = info.TrainingAccuracy;
tLoss = info.TrainingLoss;
vAcc = info.ValidationAccuracy;
vLoss = info.ValidationLoss;
xVal = 1:length(vAcc);
xVal(isnan(vAcc)) = [];
vAcc(isnan(vAcc)) = [];
vLoss(isnan(vLoss)) = [];
figure;


subplot(2,1,1);
plot(tAcc);hold on;
plot(smooth(tAcc));hold on;
plot(xVal, vAcc,'k--','LineWidth',2)

subplot(2,1,2);
plot(tLoss);hold on;
plot(smooth(tLoss)); hold on;
plot(xVal, vLoss,'k--', 'LineWidth', 2)


