warningState = warning('off','images:bigimage:singleStripTiff');
c = onCleanup(@()warning(warningState));

trainingImageDir = fullfile(tempdir,'Camelyon16','training');
if ~exist(trainingImageDir,'dir')
    mkdir(trainingImageDir);
    mkdir(fullfile(trainingImageDir,'normal'));
    mkdir(fullfile(trainingImageDir,'tumor'));
    mkdir(fullfile(trainingImageDir,'lesion_annotations'));
end

trainNormalDataDir = fullfile(trainingImageDir,'normal');
trainTumorDataDir = fullfile(trainingImageDir,'tumor');
trainTumorAnnotationDir = fullfile(trainingImageDir,'lesion_annotations');

numNormalFiles = 158;
numTumorFiles = 111; 

tumorFileName = fullfile(trainTumorDataDir,'tumor_001.tif');
tumorImage = bigimage(tumorFileName);

levelSizeInfo = table((1:length(tumorImage.LevelSizes))', ...
    tumorImage.LevelSizes(:,1), ...
    tumorImage.LevelSizes(:,2), ...
    tumorImage.LevelSizes(:,1)./tumorImage.LevelSizes(:,2), ...
    'VariableNames',["Resolution Level" "Image Width" "Image Height" "Aspect Ratio"])

h = bigimageshow(tumorImage,'ResolutionLevel',7);

xlim([29471,29763]);
ylim([117450,118110]);

h.ResolutionLevel = 1;

resolutionLevel = 7;


trainNormalMaskDir = fullfile(trainNormalDataDir,['normal_mask_level' num2str(resolutionLevel)]);
createMaskForNormalTissue(trainNormalDataDir,trainNormalMaskDir,resolutionLevel)

[bigNormalImages,bigNormalMasks] = createBigImageAndMaskArrays(trainNormalDataDir,trainNormalMaskDir);


idx = 2;
bigNormalImages(idx).SpatialReferencing(1)


bigNormalMasks(idx).SpatialReferencing(1)

figure
hBigNormal = bigimageshow(bigNormalImages(idx));
hNormalAxes = hBigNormal.Parent;

hMaskAxes = axes;
hBigMask = bigimageshow(bigNormalMasks(idx),'Parent',hMaskAxes, ...
    'Interpolation','nearest','AlphaData',0.5);
hMaskAxes.Visible = 'off';


linkaxes([hNormalAxes,hMaskAxes]);


xlim([45000 80000]);
ylim([130000 165000]);

trainTumorMaskDir = fullfile(trainTumorDataDir,['tumor_mask_level' num2str(resolutionLevel)]);
createMaskForTumorTissue(trainTumorDataDir,trainTumorAnnotationDir, ...
    trainTumorMaskDir,resolutionLevel);

[bigTumorImages,bigTumorMasks] = createBigImageAndMaskArrays(trainTumorDataDir,trainTumorMaskDir);

idx = 5;
bigTumorImages(idx).SpatialReferencing(1)

bigTumorMasks(idx).SpatialReferencing(1)

figure
hBigTumor = bigimageshow(bigTumorImages(idx));
hTumorAxes = hBigTumor.Parent;

hMaskAxes = axes;
hBigMask = bigimageshow(bigTumorMasks(idx),'Parent',hMaskAxes, ...
    'Interpolation','nearest','AlphaData',0.5);
hMaskAxes.Visible = 'off';



linkaxes([hTumorAxes,hMaskAxes]);

xlim([45000 65000]);
ylim([130000 150000]);

normalValidationIdx = randi(numNormalFiles,[1 2]);
normalTrainIdx = setdiff(1:numNormalFiles,normalValidationIdx);

patchSize = [299,299,3];

normalStrideFactor = 10;
dsNormalData = bigimageDatastore(bigNormalImages(normalTrainIdx),1, ...
    'InclusionThreshold',0.75,'Mask',bigNormalMasks(normalTrainIdx), ...
    'BlockSize',patchSize(1:2),'BlockOffsets',patchSize(1:2)*normalStrideFactor, ...
    'IncompleteBlocks','exclude');

dsNormalDataValidation = bigimageDatastore(bigNormalImages(normalValidationIdx),1, ...
    'InclusionThreshold',0.75,'Mask',bigNormalMasks(normalValidationIdx), ...
    'BlockSize',patchSize(1:2),'IncompleteBlocks','exclude');

imagesToPreview = zeros([patchSize 10],'uint8');
for n = 1:10
    im = read(dsNormalData);
    imagesToPreview(:,:,:,n) = im{1};
end
figure
montage(imagesToPreview,'Size',[2 5],'BorderSize',10,'BackgroundColor','k');
title("Training Patches of Normal Tissue")

tumorValidationIdx = randi(numTumorFiles,[1 2]);
tumorTrainIdx = setdiff(1:numTumorFiles,tumorValidationIdx);

tumorStrideFactor = 4;
dsTumorData = bigimageDatastore(bigTumorImages(tumorTrainIdx),1, ...
    'InclusionThreshold',0.75,'Mask',bigTumorMasks(tumorTrainIdx), ...
    'BlockSize',patchSize(1:2),'BlockOffsets',patchSize(1:2)*tumorStrideFactor, ...
    'IncompleteBlocks','exclude');

dsTumorDataValidation = bigimageDatastore(bigTumorImages(tumorValidationIdx),1, ...
    'InclusionThreshold',0.75,'Mask',bigTumorMasks(tumorValidationIdx), ...
    'BlockSize',patchSize(1:2),'IncompleteBlocks','exclude');

imagesToPreview = zeros([patchSize 10],'uint8');
for n = 1:10
    im = read(dsTumorData);
    imagesToPreview(:,:,:,n) = im{1};
end
montage(imagesToPreview,'Size',[2 5],'BorderSize',10,'BackgroundColor','k');
title("Training Patches of Tumor Tissue")

dsLabelledNormalData = transform(dsNormalData, ...
    @(x,info)augmentAndLabel(x,info,'normal'),'IncludeInfo',true);
dsLabelledNormalDataValidation = transform(dsNormalDataValidation, ...
    @(x,info)augmentAndLabel(x,info,'normal'),'IncludeInfo',true);


dsLabelledTumorData = transform(dsTumorData, ...
    @(x,info)augmentAndLabel(x,info,'tumor'),'IncludeInfo',true);
dsLabelledTumorDataValidation = transform(dsTumorDataValidation, ...
    @(x,info)augmentAndLabel(x,info,'tumor'),'IncludeInfo',true);

dsTrain = randomSamplingDatastore(dsLabelledTumorData,dsLabelledNormalData);

numValidationPatchesPerClass = 5;
dsValidation = validationDatastore(dsLabelledTumorDataValidation, ...
    dsLabelledNormalDataValidation,numValidationPatchesPerClass);

net = inceptionv3;


lgraph = layerGraph(net);


[learnableLayer,classLayer] = findLayersToReplace(lgraph)



numClasses = 2;
newLearnableLayer = fullyConnectedLayer(numClasses,'Name','predictions');
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);


newClassLayer = classificationLayer('Name','ClassificationLayer_predictions');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);


checkpointsDir = fullfile(trainingImageDir,'checkpoints');
if ~exist(checkpointsDir,'dir')
    mkdir(checkpointsDir);
end

options = trainingOptions('rmsprop', ...
    'InitialLearnRate',1e-5, ...
    'SquaredGradientDecayFactor',0.99, ...
    'MaxEpochs',3, ...
    'MiniBatchSize',32, ...
    'Plots','training-progress', ...
    'CheckpointPath',checkpointsDir, ... 
    'ValidationData',dsValidation, ...
    'ExecutionEnvironment','auto', ...
    'Shuffle','every-epoch');


testingImageDir = fullfile(tempdir,'Camelyon16','testing');
if ~exist(testingImageDir,'dir')
    mkdir(testingImageDir);
    mkdir(fullfile(testingImageDir,'images'));
    mkdir(fullfile(testingImageDir,'lesion_annotations'));
end

testDataDir = fullfile(testingImageDir,'images');
testTumorAnnotationDir = fullfile(testingImageDir,'lesion_annotations');

numTestFiles = 2;

resolutionLevel = 7;

testTissueMaskDir = fullfile(testDataDir,['test_tissuemask_level' num2str(resolutionLevel)]);
createMaskForNormalTissue(testDataDir,testTissueMaskDir,resolutionLevel);

testTumorMaskDir = fullfile(testDataDir,['test_tumormask_level' num2str(resolutionLevel)]);
createMaskForTumorTissue(testDataDir,testTumorAnnotationDir,testTumorMaskDir,resolutionLevel);

doTraining = false;
if doTraining
    trainedNet = trainNetwork(dsTrain,lgraph,options);
    save(['trainedCamelyonNet-' datestr(now, 'dd-mmm-yyyy-HH-MM-SS') '.mat'],'trainedNet');
else
    trainedCamelyonNet_url = 'https://www.mathworks.com/supportfiles/vision/data/trainedCamelyonNet.mat';
    netDir = fullfile(tempdir,'Camelyon16');
    downloadTrainedCamelyonNet(trainedCamelyonNet_url,netDir);
    load(fullfile(netDir,'trainedCamelyonNet.mat'));
end


[bigTestImages,bigTestTissueMasks] = createBigImageAndMaskArrays(testDataDir,testTissueMaskDir);
[~,bigTestTumorMasks] = createBigImageAndMaskArrays(testDataDir,testTumorMaskDir);

patchSize = [299,299,3];

for idx = 1:numTestFiles
    bigTestHeatMaps(idx) = apply(bigTestImages(idx),1,@(x)createHeatMap(x,trainedNet), ...
        'Mask',bigTestTissueMasks(idx),'InclusionThreshold',0, ...
        'BlockSize',patchSize(1:2),'UseParallel',false);
    
    figure
    hBigTest = bigimageshow(bigTestImages(idx));
    hTestAxes = hBigTest.Parent;
    hTestAxes.Visible = 'off';
        
    hMaskAxes = axes;
    hBigMask = bigimageshow(bigTestHeatMaps(idx),'Parent',hMaskAxes, ...
        "Interpolation","nearest","AlphaData",bigTestTissueMasks(idx));
    colormap(jet(255));
    hMaskAxes.Visible = 'off';
    
    linkaxes([hTestAxes,hMaskAxes]);
    title(['Tumor Heatmap of Test Image ',num2str(idx)])
end




