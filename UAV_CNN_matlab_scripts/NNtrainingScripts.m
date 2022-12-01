
%% create an image data store
% tip = 'D:\NAI\Training\ImageData\27cm\DEM';
tip = '/Volumes/Valaquenta/NAI/Training/ImageData/6cm/Vis/';
imds = imageDatastore(tip);

%% create class vector
classes = {'BorderPixels', 'PolygonRidge', 'AeolianCover',...
    'MottledGround', 'Road', 'ErodedRidgesAndTumuli',...
    'Tumulus',  'Objects', 'PatternedGround',...
    'DrainageChannelRidge', 'MudCrack', 'SaltPan'};
%% create a pixel label data store
% ldp = 'D:\NAI\Training\LabelData\27cm\DEM';
ldp = '/Volumes/Valaquenta/NAI/Training/LabelData/6cm/Vis';
pxds = pixelLabelDatastore(ldp,classes,0:11);

%% create weights for each class for training
tbl = countEachLabel(imdsTest);
freq = tbl.PixelCount/sum(tbl.PixelCount);
imFreq = tbl.PixelCount./tbl.ImagePixelCount;
classWeights = median(imFreq)./imFreq;
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
%plot labels
figure;
bc = bar(1:numel(classes),freq,'FaceColor','flat');
bc.CData = cMap;
xticks(1:numel(classes))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')
xlabel('Classes')
title('Breakdown of Classes in Test Dataset')

%% Load neural net
imageSize = [224 224 3];
netname = 'resnet50';
lgraph = deeplabv3plusLayers(imageSize, numel(classes),netname);
fprintf('Layer graph of %s uploaded\n\n',netname)

%% modify the network by freezing initial layers and replacing classifying layers

%--------------------------------------------------------------
% add paths that you need
% addpath(genpath('C:\Program Files\MATLAB\R2020a\examples'))
% addpath(genpath('D:\NAI'))
%--------------------------------------------------------------

%--------------------------------------------------------------
% freeze or slow the first 15 layers
% for resnet, there are 5 convolutions in the first 15 layers

%--------------------------------------------------------------
% for RGB-Network
%--------------------------------------------------------------
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(2:16)=adjustWeights(layers(2:16), 0);%adjust RGB first layer group
lgraph = createLgraphUsingConnections(layers,connections);
%--------------------------------------------------------------



imInputLayer = imageInputLayer([224 224 3],'Name','InputLayer');

% replace classification layers of lgraph for transfer learning
pxLayer = pixelClassificationLayer('Name','labels','Classes',classes,...
    'ClassWeights',classWeights);

convFinal = convolution2dLayer([1,1],numel(classes),...
    'Name','Scorer',...
    'NumChannels','auto',...
    'Stride',[1,1],'DilationFactor',[1,1],...
    'Padding','same','WeightLearnRateFactor',100,...
    'BiasLearnRateFactor',100);
 
TconvFinal = transposedConv2dLayer([8,8],numel(classes),...
    'Name','dec-Upsample2',...
    'NumChannels',numel(classes),...
    'Stride',[4,4],'Cropping',2,...
    'WeightLearnRateFactor',100,...
    'BiasLearnRateFactor',100);

% replace layers
%--------------------------------------------------------------
% for RGB-Network or DEM-Network
%--------------------------------------------------------------
ScorerName = lgraph.Layers(202,1).Name; 
TconName = lgraph.Layers(203,1).Name;
%--------------------------------------------------------------


inName = lgraph.Layers(1,1).Name;
pxName = lgraph.Layers(end,1).Name;
lgraph = replaceLayer(lgraph,inName,imInputLayer);
lgraph = replaceLayer(lgraph,ScorerName,convFinal);
lgraph = replaceLayer(lgraph,TconName,TconvFinal);
lgraph = replaceLayer(lgraph,pxName,pxLayer);

fprintf('Layers replaced\n\n')

analyzeNetwork(lgraph);

%% data prep
% Define training options. 
%     'Momentum',0.9,...

% augmenter = imageDataAugmenter('RandXReflection',true,...
%     'RandYReflection',true,...
%     'RandRotation',[-180 180],...
%     'RandXTranslation',[-50 50], 'RandYTranslation',[-50 50]);
% 

pximds = pixelLabelImageDatastore(imds,pxds);%,...
%     'DataAugmentation',augmenter);

%%
% indeces = randperm(pximds.NumObservations);
% trainIndex = indeces(1:pximds.NumObservations*0.75);
% valIndex = indeces(pximds.NumObservations*0.75+1:...
%     pximds.NumObservations*0.75+pximds.NumObservations*0.125);
% testIndex = indeces(pximds.NumObservations*0.75+pximds.NumObservations*0.125+1:...
%     end);


%% Train
mbs = 16;%minibatch size
options = trainingOptions('adam', ...
    'ValidationFrequency',floor(length(indx{1}*6)/mbs),...
    'ValidationData',imdsVal,...
    'ValidationPatience',3,...
    'InitialLearnRate', 1e-4,...
    'LearnRateSchedule','piecewise',...
    'MaxEpochs',50,...  
    'MiniBatchSize',mbs,...
    'Shuffle','every-epoch',...
    'VerboseFrequency',5,...
    'OutputNetwork','best-validation-loss',...
    'Plots','training-progress');

[RGBNet6p9cmE1to50, RGBNet6p9cmInfoE1to50] = trainNetwork(imdsTrain,lgraph,options);

%% keep training? 
% augmenter = imageDataAugmenter('RandXReflection',true,...
%     'RandYReflection',true,...
%     'RandXTranslation',[-90 90],'RandYTranslation',[-20 20],...
%     'RandRotation',[-180 180]);
% 
% pximds = pixelLabelImageDatastore(imds,pxds, ...
%     'DataAugmentation',augmenter);

%     'LearnRateDropFactor',0.2,...

opts2 = options;
% opts2 = trainingOptions('sgdm', ...
%     'LearnRateSchedule','piecewise',...
%     'InitialLearnRate',5e-3, ...
%     'Momentum',0.4,...
%     'LearnRateDropFactor',0.1,...
%     'LearnRateDropPeriod',10,...
%     'L2Regularization',0.05, ...
%     'MaxEpochs',8, ...  
%     'MiniBatchSize',15, ...
%     'Shuffle','every-epoch', ...
%     'VerboseFrequency',1,...
%     'Plots','training-progress');

%% evaluate networks

pxdsResults = semanticsegMultiInputNet(imdsTest,RgbDemNet30p9cm);
% pxdsResults = semanticseg(imdsTest,DEMNet30p9cmE1to50);
%%
RGBNet6p9cmSSM = evaluateSemanticSegmentation(pxdsResults,imdsTest.UnderlyingDatastores{3});

