% calculate class uncertainty
srce = 'D:\NAI\Training\Networks\TrainedNetworks\Metrics\';
dirs = {'3cm','6p9cm','10p3cm','13p7cm',...
    '17p1cm','20p5cm','23p9cm','27p3cm','30p9cm'};
netTypes = {'RGB','DEM','RgbDem'};
classes = {'BorderPixels', 'PolygonRidge', 'AeolianCover',...
    'MottledGround', 'Road', 'ErodedRidgesAndTumuli',...
    'Tumulus',  'Objects', 'PatternedGround',...
    'DrainageChannelRidge', 'MudCrack', 'SaltPan'};


for i = length(dirs)
    for l = 1:length(netTypes)
        
        if i == length(dirs)
            if l == 3
                continue
            end
            load([srce,dirs{i},'\',netTypes{l},'Net30p9cmResults.mat']);
            classUC = zeros(length(R),length(classes));
            
            for k = 1:length(R)
                UC = R{k,3};
                predict = R{k,5};
            
                for c = 1:length(classes)
                    predictMask = zeros(size(predict));
                    [pr,pc] = find(predict==classes(c));
                        for j = 1:length(pr)
                            predictMask(pr(j),pc(j)) = 1;
                        end
                    tpMask = UC.*predictMask;
                    classUC(k,c) = sum(sum(tpMask))./sum(sum(predictMask));

                end
            end
            
            avgClassUC = mean(classUC,1,'omitnan');
            save([srce,dirs{i},'\',netTypes{l},'\',netTypes{l},dirs{i},'classUC.mat'],'classUC','avgClassUC');
            clear classUC;
            clear avgClassUC;
        
        else
            load([srce,dirs{i},'\',netTypes{l},'\Prediction\Predictions.mat']);
            classUC = zeros(length(Parray),length(classes));

            if i==1 && l==3
                continue
            end

            for k = 1:length(Parray)
                UCName = [srce,dirs{i},'\',netTypes{l},'\','Uncertainty\Uncertainty',num2str(k),'.bsq'];
                pxNum = length(Parray{k});
                UC = multibandread(UCName,[pxNum,pxNum,1],'single',0,'bsq','ieee-le');
                predict = Parray{k};

                for c = 1:length(classes)
                    predictMask = zeros(size(predict));
                    [pr,pc] = find(predict==classes(c));
                    for j = 1:length(pr)
                        predictMask(pr(j),pc(j)) = 1;
                    end
                    tpMask = UC.*predictMask;
                    classUC(k,c) = sum(sum(tpMask))./sum(sum(predictMask));

                end
            end
            avgClassUC = mean(classUC,1,'omitnan');
            safa([srce,dirs{i},'\',netTypes{l},'\',netTypes{l},dirs{i},'classUC.mat'],'classUC','avgClassUC');
            clear classUC;
            clear avgClassUC;
        end
        clear Parray
    end
end

            