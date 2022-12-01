% load data
% -------------------------------------------------------
% PC
% addpath(genpath('D:\NAI\'))
% load('D:\NAI\Training\Networks\RGBNetwork3cm.mat');
% clear lgraph
% tip = 'D:\NAI\Training\ImageData\27cm\Vis';
% ldp = 'D:\NAI\Training\LabelData\27cm\Vis';
% -------------------------------------------------------
% Mac
clear all;close all;clc
addpath(genpath('/Volumes/Valaquenta/NAI/Training'))
load('/Volumes/Valaquenta/NAI/Training/Networks/RGBNetwork6p9cm.mat');
tip = '/Volumes/Valaquenta/NAI/Training/ImageData/6cm/Vis';
ldp = '/Volumes/Valaquenta/NAI/Training/LabelData/6cm/Vis';
clear lgraph
% -------------------------------------------------------
imds = imageDatastore(tip);
pxds = pixelLabelDatastore(ldp,classes,0:11);
pximds = pixelLabelImageDatastore(imds,pxds);
imdsTest = partitionByIndex(pximds,testIndex);

%% load net
% -------------------------------------------------------
% PC
% load('D:\NAI\Training\Networks\TrainedNetworks\RGBNet30p9cmE1to70.mat','RGBNet30p9cmDO');
% -------------------------------------------------------
% Mac
load('/Volumes/Valaquenta/NAI/Training/Networks/TrainedNetworks/RGB/RGBNet30p9cmE1to50.mat','RGBNet30p9cmE1to50');
% -------------------------------------------------------
net = RGBNet30p9cmE1to50;

%% calculate results
% response = imdsTest.readall;
% name = net.Layers(210).Name;
% N = 100;
% n = 20;
% AverageImage = 0;
% Y = single(zeros([size(response.pixelLabelImage{1}),length(classes),N]));
% avgIm = single(zeros([size(response.pixelLabelImage{1}),length(classes),N]));
% varIm = single(zeros([size(response.pixelLabelImage{1}),length(classes),N]));
% RMSE = zeros([size(response.pixelLabelImage{1}),length(classes)]);
% rmse = zeros([size(response.pixelLabelImage{1}),length(classes),N]);
% sqrs = zeros([size(response.pixelLabelImage{1}),length(classes),N]);
% SS = zeros([size(response.pixelLabelImage{1}),length(classes)]);
% P = categorical(zeros(size(response.pixelLabelImage{1})));
% P = setcats(P,classes);
% R = cell(imdsTest.NumObservations,5);
% 

response = imdsTest.readall;
name = net.Layers(209).Name;
N = 100;
n = 20;
Y = single(zeros([size(response.pixelLabelImage{1}),length(classes),N]));
avgIm = single(zeros([size(response.pixelLabelImage{1}),length(classes),N]));
varIm = single(zeros([size(response.pixelLabelImage{1}),length(classes),N]));
RMSE = zeros([size(response.pixelLabelImage{1}),length(classes)]);
rmse = zeros([size(response.pixelLabelImage{1}),length(classes),N]);
sqrs = zeros([size(response.pixelLabelImage{1}),length(classes),N]);
SS = zeros([size(response.pixelLabelImage{1}),length(classes)]);
P = categorical(zeros(size(response.pixelLabelImage{1})));
P = setcats(P,classes);
R = cell(imdsTest.NumObservations,5);
res = '30p9cm';
savePath = ['/Volumes/Valaquenta/NAI/Training/Networks/TrainedNetworks/Metrics/',res,'/RGB/'];

% Parray = cell(imdsTest.NumObservations,1);

U = zeros(size(response.pixelLabelImage{1}));


numPxls = size(response.pixelLabelImage{1},1);
tic

inImages = response.inputImage;
for j = 151:imdsTest.NumObservations
    
    inImage = inImages{j};
%     if size(inImage) ~= size(Y(:,:,:,1))
%         a = 'stp';
%     end
    parfor i = 1:N
%         Y(:,:,:,i) = net.activations(imImage, name);
        Y(:,:,:,i) = activations(net,inImage, name);
    end
    
    AverageImage = mean(Y,4,'omitnan');
    VarianceImage = var(Y,0,4,'omitnan');
    for i = 1:numPxls
%     parfor i = 1:numPxls
        for k = 1:numPxls
            AvgClasses = AverageImage(i,k,:);
            VarClasses = VarianceImage(i,k,:);
            mi = find(AvgClasses==max(AvgClasses));
            P(i,k) = classes(mi(1));
            
            AvgMax = AvgClasses(mi(1));
            VMax = VarClasses(mi(1));
            Confidence = zeros([1,length(classes)]);
            T = zeros([1,length(classes)]);
            for z = 1:length(classes)
                
                if z == mi
                    Confidence(z) = NaN;
                    T(z) = NaN;
                    continue
                end
                
                T(z) = (AvgMax-AvgClasses(z))./((VMax + VarClasses(z)).^0.5);
                tcrit = T(z);
                f = @(prob) tcrit - tinv(prob,99);
%                 Confidence(z) = cFun(z,f);
                try
                    Confidence(z) = fzero(f,[0.0001 0.9999]);
                catch e
                    Confidence(z) = 0.9999;
                end   
                if i ==12 && k ==13
                    stop = 1;
                end
            end
            U(i,k) = (min(Confidence)*100-50)*2;
        end
    end
    for i = 1:N
        sqrs(:,:,:,i) = (Y(:,:,:,i)-AverageImage).^2;
        rmse(:,:,:,i) = (sqrs(:,:,:,i)).^0.5;
    end
    RMSE = mean(rmse,4,'omitnan');
    SS  = sum(sqrs,4);
    
%     multibandwrite
    Parray{j} = P;
    AvgName = [savePath,'/AverageImage/Avg',num2str(j),'.bsq'];
    VarName = [savePath,'/VarianceImage/Var',num2str(j),'.bsq'];
    UCName = [savePath,'/Uncertainty/Uncertainty',num2str(j),'.bsq'];
    PName = [savePath,'/Prediction/Predictions.mat'];
    multibandwrite(AverageImage,AvgName,'bsq');
    multibandwrite(VarianceImage,VarName,'bsq');
    multibandwrite(U,UCName,'bsq');
    if j==1
        save(PName,'Parray');
    else
        save(PName,"-append",'Parray');
    end


%     R{j,1} = AverageImage;
%     R{j,2} = VarianceImage;
%     R{j,3} = U;
%     R{j,4} = RMSE;
%     R{j,5} = P;
    if j == imdsTest.NumObservations
        continue
    end
%     clear AverageImage VarianceImage Y rmse LL RMSE SS sqrs
    fprintf('Processed image %d of %d\n',j,imdsTest.NumObservations);
%     if j == 3
%         stop = 1;
%     end
    
end
toc


% for i = 1:length(R);R{i,5}=setcats(R{i,5},classes);end
%% load R
% Use this when you don't calculate the results using the block above
res = '30p9cm';
s = size(imds.read());
R = cell(imdsTest.NumObservations,3);
% A = imageDatastore(['/Volumes/Valaquenta/NAI/Training/Networks/TrainedNetworks/Metrics/',res,'/RGB/AverageImage/'],'FileExtensions','.bsq');
% V = imageDatastore(['/Volumes/Valaquenta/NAI/Training/Networks/TrainedNetworks/Metrics/',res,'/RGB/VarianceImage/'],'FileExtensions','.bsq');
% U = imageDatastore(['/Volumes/Valaquenta/NAI/Training/Networks/TrainedNetworks/Metrics/',res,'/RGB/Uncertainty/'],'FileExtensions','.bsq');
response = imdsTest.readall;
P = categorical(zeros(size(response.pixelLabelImage{1})));
P = setcats(P,classes);
f = 'Fold1';
load(['/Volumes/Valaquenta/NAI/Training/Networks/TrainedNetworks/Metrics/',res,'/RGB/',f,'/Prediction/Predictions_Fold1.mat'],'Parray');
Parray2 = Parray;
for i = 1:length(imdsTest.Images)
    afile = ['/Volumes/Valaquenta/NAI/Training/Networks/TrainedNetworks/Metrics/',res,'/RGB/',f,'/AverageImage/Avg',num2str(i),'.bsq'];
    vfile = ['/Volumes/Valaquenta/NAI/Training/Networks/TrainedNetworks/Metrics/',res,'/RGB/',f,'/VarianceImage/Var',num2str(i),'.bsq'];
    ufile = ['/Volumes/Valaquenta/NAI/Training/Networks/TrainedNetworks/Metrics/',res,'/RGB/',f,'/Uncertainty/Uncertainty',num2str(i),'.bsq'];
    R{i,1} = multibandread(afile,[s(1),s(1),12],'single',0,'bsq','ieee-le');
    R{i,2} = multibandread(vfile,[s(1),s(1),12],'single',0,'bsq','ieee-le');
    R{i,3} = multibandread(ufile,[s(1),s(1),1],'single',0,'bsq','ieee-le');
%     R{i,4} = Parray{i};
%     AverageImage = mean(Y,4,'omitnan');
%     VarianceImage = var(Y,0,4,'omitnan');
%     for j = 1:s(1)
% %     parfor i = 1:numPxls
%         for k = 1:s(2)
%             AvgClasses = R{i}(j,k,:);
%             mi = find(AvgClasses==max(AvgClasses));
%             P(j,k) = classes(mi(1));
%         end
%     end
%     Parray{i} = P;
end
%% calculate metrics
% load netwok
%load('D:\NAI\Training\Networks\TrainedNetworks\RGBNet3cmE1to30.mat','RGBNet3cmE1to30');
%lgraph = RGBNet3cmE1to30;

% segment images
% cd D:\NAI\Training\TestResults\30p9cm\DropOut\RGBDO3
% pxdsResults = semanticseg(imdsTest,RGBNet30p9cm4);
% pxdsResults = pixelLabelDatastore('./',classes,0:11);

% calculate metrics
GSD = 6.9;
im_diag = (size(R{1,1},1).^2 + size(R{1,1},2).^2).^0.5;
PRscore = cell(length(R));
PRprecision = cell(length(R),1);
PRrecall = cell(length(R),1);
Dscore = cell(length(R),1);
Dprecision = cell(length(R),1);
Drecall = cell(length(R),1);
PGscore = cell(length(R),1);
PGprecision = cell(length(R),1);
PGrecall = cell(length(R),1);
PRsimilarity = cell(length(R),1);
Dsimilarity = cell(length(R),1);
PGsimilarity = cell(length(R),1);
% response = imdsTest.readall;
for i = 1:length(R)
%     prediction = pxdsResults.readimage(i);
    prediction = Parray{i};
%     prediction = R{i,5};
%     prediction = R{i,4}; %for when loading R manually
    r = response.pixelLabelImage{i};
    t = 1;
       
%     22.9cm
%     t = 22.9/GSD;
    [PRscore{i}, PRprecision{i}, PRrecall{i}] = ...
        bfscore(prediction,r,t);
%     [PRscore{i}, PRprecision{i}, PRrecall{i}] = ...
%         bfscore(prediction,r);
    PRsimilarity{i} = jaccard(prediction,r);   
%     t = 16.1/GSD;
    [Dscore{i}, Dprecision{i}, Drecall{i}] = ...
        bfscore(prediction,r,t);
%     [Dscore{i}, Dprecision{i}, Drecall{i}] = ...
%         bfscore(prediction,r);
    Dsimilarity{i} = jaccard(prediction,r);   
%     t = 11.7/GSD;
    [PGscore{i}, PGprecision{i}, PGrecall{i}] = ...
        bfscore(prediction,r,t);
%     [PGscore{i}, PGprecision{i}, PGrecall{i}] = ...
%         bfscore(prediction,r);
    PGsimilarity{i} = jaccard(prediction,r);   
end

PRScoreTbl = ones(12,length(PRscore));
DScoreTbl = ones(12,length(PRscore));
PGScoreTbl = ones(12,length(PRscore));

PRPrecTbl = ones(12,length(PRscore));
DPrecTbl = ones(12,length(PRscore));
PGPrecTbl = ones(12,length(PRscore));

PRRecallTbl = ones(12,length(PRscore));
DRecallTbl = ones(12,length(PRscore));
PGRecallTbl = ones(12,length(PRscore));

PRjaccardTbl = ones(12,length(PRscore));
DjaccardTbl = ones(12,length(PRscore));
PGjaccardTbl = ones(12,length(PRscore));

for i = 1:length(PRscore)
    PRScoreTbl(:,i) = PRscore{i};
    DScoreTbl(:,i) = Dscore{i};
    PGScoreTbl(:,i) = PGscore{i};
    
    PRPrecTbl(:,i) = PRprecision{i};
    DPrecTbl(:,i) = Dprecision{i};
    PGPrecTbl(:,i) = PGprecision{i};
    
    PRRecallTbl(:,i) = PRrecall{i};
    DRecallTbl(:,i) = Drecall{i};
    PGRecallTbl(:,i) = PGrecall{i};

    PRjaccardTbl(:,i) = PRsimilarity{i};
    DjaccardTbl(:,i) = Dsimilarity{i};
    PGjaccardTbl(:,i) = PGsimilarity{i};
end

PRMBF = mean(PRScoreTbl(2,:),'omitnan');
DMBF  = mean(DScoreTbl(7,:),'omitnan');
PGMBF = mean(PGScoreTbl(9,:),'omitnan');

PRprec = mean(PRPrecTbl(2,:),'omitnan');
Dprec  = mean(DPrecTbl(7,:),'omitnan');
PGprec = mean(PGPrecTbl(9,:),'omitnan');

PRRec = mean(PRRecallTbl(2,:),'omitnan');
DRec  = mean(DRecallTbl(7,:),'omitnan');
PGRec = mean(PGRecallTbl(9,:),'omitnan');

PRJac = mean(PRjaccardTbl(2,:),'omitnan');
DJac  = mean(DjaccardTbl(7,:),'omitnan');
PGJac = mean(PGjaccardTbl(9,:),'omitnan');

% metrics = [PRMBF,PRprec,PRRec;DMBF,Dprec,DRec;PGMBF,PGprec,PGRec];
% metrics = array2table(metrics);
% metrics.Properties.RowNames = {'PolygonRidge','Dome','PatternedGround'};
% metrics.Properties.VariableNames={'BF-Score','Precision','Recall'};
% 
% disp(metrics);

metrics2 = [PRMBF,PRprec,PRRec,PRJac;DMBF,Dprec,DRec,DJac;PGMBF,PGprec,PGRec,PGJac];
metrics2 = array2table(metrics2);
metrics2.Properties.RowNames = {'PolygonRidge','Dome','PatternedGround'};
metrics2.Properties.VariableNames={'BF-Score','Precision','Recall','IoU'};

disp(metrics2);

%% uncertainty analysis
% U = zeros([60 60]);
% % C = categorical(zeros([60 60]));
% T = zeros([1,length(classes)]);
% % imInd = 134;
% % Confidence = zeros([1,length(classes)]);
% tic
% for q = 1%1:length(R)
%     for j = 1:60
%         for k = 1:60
%     %         LLim = R{imInd,3}(j,k,:);
%             Avg  = R{q,1}(j,k,:);
%             V       = R{q,2}(j,k,:);
% 
%             mi = find(Avg==max(Avg));
%     %         LLmi = find(LLim==max(LLim));
%     %         C(j,k) = categorical(mi);
%     %         C(j,k) = classes(LLmi);
%             for i = 1:length(classes)
%                 if i == mi
%                     T(i) = NaN;
%                     Confidence(i) = NaN;
%                     continue
%                 end
%                 T(i) = (Avg(mi)-Avg(i))/((V(mi)/N + V(i)/N)^0.5);
%                 tcrit = T(i);
%                 f = @(prob) tcrit - tinv(prob,399);% - tcrit;
%                 try
%                     Confidence(i) = fzero(f,[0.0001 0.9999]);
%                 catch e
%                     Confidence(i) = 0.9999;
%                 end            
%             end
%             U(j,k) = (min(Confidence)*100-50)*2;
%         end
%     end
%     R{q,3} = U;
%     fprintf('Processed Image %d of %d\n',q,length(R));
% end
% 
% toc


%% plot all the things
% addpath('C:\Users\Michael\Documents\MATLAB\subplot_tight\')
addpath(genpath('/Users/phillms1/Documents/MATLAB/subtightplot/'));
cMap = [0 0 0;
    255/255 115/255 223/255;
    0 197/255 255/255;
    55/255 108/255 189/255;
    25/255 25/255 25/255;
    240/255 204/255 230/255;
    181/255 53/255 53/255;
    50/255 50/255 50/255;
    85/255 255/255 0/255;
    231/255 216/255 240/255;
    190/255 232/255 255/255;
    216/255 240/255 231/255];
% N = [31,80,87,16];
N = 151;
for g = 1:length(N)
    n = N(g);
    c = 7;
    c2 = 2;
    % 
    % A = labeloverlay(p.inputImage{n},R{n,1}(:,:,2));%,'Colormap',jet,'Transparency',0.5);
    % V = labeloverlay(p.inputImage{n},R{n,2}(:,:,2));%,'Colormap',jet,'Transparency',0.5);
    
    b = labeloverlay(response.inputImage{n},response.pixelLabelImage{n},'Colormap',cMap,'Transparency',0.8);
%     b = labeloverlay(p.inputImage{n},C,'Colormap',cMap,'Transparency',0.80);
%     prediction = labeloverlay(response.inputImage{n},pxdsResults.readimage(n),'Colormap',cMap,'Transparency',0.80);
    prediction = labeloverlay(response.inputImage{n},Parray{N},'colormap', cMap, 'Transparency',0.8);

    % 3 image plot

    figure
    subplot(2,3,1)
%     imshow(p.inputImage{n})
    imshow(b)
%     imagesc(R{N,3})
%     title('Confidence')
    axis square
%     colorbar
%     set(gca,'Colormap',jet)
    title('Ground Truth')
%     title('Statistical Prediction')
    colormap(gca,cMap);cbar=colorbar('peer',gca);cbar.TickLabels=classes;
    cbar.Ticks = 1/(numel(classes)*2):1/numel(classes):1;
    cbar.TickLength = 0;

    subplot(2,3,2)
%     imagesc(R{n,3})
    imagesc(R{n,1}(:,:,c))
    set(gca,'ColorMap',jet)
    colorbar
    axis square
    axis off
%     title('Certainty (0 to 1)')
    title('Dome Average')
    
    subplot(2,3,3)
    imagesc(R{n,2}(:,:,c))
    set(gca,'ColorMap',jet)
    colorbar
    axis square
    axis off
    title('Dome Variance') 
    
    subplot(2,3,4)
    imshow(prediction)
    title('Prediction')
%     title('One-off Prediction')
    colormap(gca,cMap);cbar=colorbar('peer',gca);cbar.TickLabels=classes;
    cbar.Ticks = 1/(numel(classes)*2):1/numel(classes):1;
    cbar.TickLength = 0;
    
    subplot(2,3,5)
    imagesc(R{n,1}(:,:,c2))
    set(gca,'ColorMap',jet)
    colorbar
    axis off
    axis square
    title('PolygonRidge Average')
    
    subplot(2,3,6)
    imagesc(R{n,2}(:,:,c2))
    set(gca,'ColorMap',jet)
    colorbar
    axis square
    axis off
    title('PolygonRidge Variance')

%     sgtitle(sprintf('Test Image %d',n))

end
%6 image plot
% figure
% subplot(2,3,1)
% imshow(p.inputImage{n});
% subplot(2,3,2)
% imagesc(R{n,1}(:,:,2))
% set(gca,'ColorMap',jet)
% axis square
% subplot(2,3,3)
% imagesc(R{n,2}(:,:,2))
% set(gca,'ColorMap',jet)
% axis square
% subplot(2,3,4)
% imshow(b)
% subplot(2,3,5)
% imshow(imadjust(R{n,1}(:,:,[2,6,7]),stretchlim(R{n,1}(:,:,[2,6,7])),[0.001 0.999]));
% subplot(2,3,6)
% imshow(imadjust(R{n,2}(:,:,[2,6,7]),stretchlim(R{n,2}(:,:,[2,6,7])),[0.001 0.999]));
%% visualize data

figure
montage(response.inputImage(101:200))
%% visualize activations
cMap = [0 0 0;
    255/255 115/255 223/255;
    0 197/255 255/255;
    55/255 108/255 189/255;
    25/255 25/255 25/255;
    240/255 204/255 230/255;
    181/255 53/255 53/255;
    50/255 50/255 50/255;
    85/255 255/255 0/255;
    231/255 216/255 240/255;
    190/255 232/255 255/255;
    216/255 240/255 231/255];

l13 = labeloverlay(response.inputImage{13},response.pixelLabelImage{13},'Transparency',0.85,'Colormap',cMap);
l16 = labeloverlay(response.inputImage{31},response.pixelLabelImage{31},'Transparency',0.85,'Colormap',cMap);
l23 = labeloverlay(response.inputImage{23},response.pixelLabelImage{23},'Transparency',0.85,'Colormap',cMap);
Y13 = Y(:,:,[2,7,4],13);
Y16 = Y(:,:,[2,7,10],31);
Y23 = Y(:,:,[2,7,6],23);

m13 = 255./(max(max(Y13))-min(min(Y13)));
m16 = 255./(max(max(Y16))-min(min(Y16)));
m23 = 255./(max(max(Y23))-min(min(Y23)));
b13 = 0-m13.*min(min(Y13));
b16 = 0-m16.*min(min(Y16));
b23 = 0-m23.*min(min(Y23));

Y13 = uint8(Y13.*m13+b13);
Y16 = uint8(Y16.*m16+b16);
Y23 = uint8(Y23.*m23+b23);

figure;
montage({Y13(:,:,:),l13,Y16(:,:,:),l16,Y23(:,:,:),l23},'Size',[3,2])

%% calculate some probabilities for each class
%crop each image if neccessary and put into cell array
for i = 1:length(Y)
    A{i} = Y(:,:,:,i); %make sure you change this if you need to crop at all
end
%% calculate class-wise means

z = zeros(size(response.pixelLabelImage{1}));
for i = 1:length(response.pixelLabelImage)
    % create masks
    for j = 1:length(classes)
%       create masks
        cMask{i,j} = z;
        cMask{i,j}(response.pixelLabelImage{i}==classes(j))=1;
        cMask{i,j}(cMask{i,j}==0)=NaN;
        
%         multiply activations by masks, activations are for image i and
%         class mask j
        aMask{i,j} = A{i}.*cMask{i,j};
        for k = 1:length(classes)
            %mean for class k, mask j, image i
            cMeans(k,j,i) = nanmean(reshape((aMask{i,j}(:,:,k)),...
                [numel(aMask{i,j}(:,:,k)),1]));
            cStDs(k,j,i) = nanstd(reshape((aMask{i,j}(:,:,k)),...
                [numel(aMask{i,j}(:,:,k)),1]));
        end
    end
end


mMeans  = nanmean(cMeans,3);
sStds   = nanmean(cStDs,3);

results = zeros([length(mMeans),2*length(mMeans)]);
results(:,1:2:end) = mMeans;
results(:,2:2:end) = sStds;


%% calculate a total confidence score for each class

ConfidenceScore = zeros(length(response.inputImage),length(classes));
% create masks
testTable = imdsTest.readall;
for i = 1:length(testTable.inputImage)
    gt = testTable.pixelLabelImage{i};
    predict = R{i,5};
    U = R{i,3};
    for c = 1:length(classes)
        gtMask = zeros(size(gt));
        predictMask = zeros(size(gt));
        [gtr,gtc] = find(gt==classes(c));
        [pr,pc] = find(predict==classes(c));
        for j = 1:length(gtr)
            gtMask(gtr(j),gtc(j))=1;
        end
        for j = 1:length(pr)
            predictMask(pr(j),pc(j)) = 1;
        end
        tpMask = gtMask.*predictMask;
        fpMask = predictMask-gtMask;
        fpMask(fpMask==-1)=0;
        tpU = tpMask.*U;
        tpU(tpU==0)=[];
        PC = mean(tpU,'all','omitnan');
        fpU = fpMask.*U;
        fpU(fpU==0)=[];
        NC = mean(fpU,'all','omitnan');
        numGuesses = sum(predictMask,'all','omitnan');
        numTP = sum(tpMask,'all','omitnan');
        numFP = sum(fpMask,'all','omitnan');
        if numGuesses > 0 && numTP > 0 && numFP >0
            ConfidenceScore(i,c) = numTP.*PC./numGuesses + ...
                numFP.*(100-NC)./numGuesses;
        elseif numTP == 0 && numFP >0
            ConfidenceScore(i,c) = 100-NC;
        elseif numTP > 0 && numFP ==0
            ConfidenceScore(i,c) = PC;
        else
            ConfidenceScore(i,c) = numTP.*PC./numGuesses + ...
                numFP.*(100-NC)./numGuesses;
        end
            
        check = 1;
    end
        check = 1;
end

ClassConfidence = mean(ConfidenceScore,1,'omitnan')';
disp(ClassConfidence)

%% plot activation pdfs

prow = 12;
pcol = 13;
y = cell(12,1);
x = cell(12,1);
for i =1:12
    x{i}=zeros(100,1);
    x{i}(:,:) = Y(prow,pcol,i,:);
    y{i} = makedist('Normal','mu',AverageImage(prow,pcol,i),...
        'sigma',VarianceImage(prow,pcol,i)^0.5);
end

figure;
for i = 1:length(classes)%[2,3,7,9]
    histogram(x{i},10,'Normalization','pdf','FaceColor',cMap(i,:));hold on;
    stop = 1;
end
for i = 1:length(classes)%[2,3,7,9]
    xdat = (min(x{i})-0.1*abs(min(x{i}))):0.1:(max(x{i})+0.1*abs(max(x{i})));
    pd=pdf(y{i},xdat);
    plot(xdat,pd,'Color',cMap(i,:),'LineWidth',3);hold on;
end
legend(classes)
title('PDFs for a high confidence pixel')
xlabel('Activation Values')



%%
function Confidence = cFun(z, f)
    try
        Confidence(z) = fzero(f,[0.0001 0.9999]);
    catch e
        Confidence(z) = 0.9999;
    end       
end