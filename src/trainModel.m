clear; clc;
load emotionFeatures_sequential.mat;

% --- 1. Create Data Partition (Manual Method) ---
numSamples = numel(labels);
rng(123); % For reproducible results
idx = randperm(numSamples);
splitPoint = floor(0.80 * numSamples);

idxTrain = idx(1:splitPoint);
idxValidation = idx(splitPoint+1:end);

XTrain = features(idxTrain);
YTrain = labels(idxTrain);
XValidation = features(idxValidation);
YValidation = labels(idxValidation);

% --- 2. Normalize Features ---
allTrainFrames = cat(2, XTrain{:});
mu = mean(allTrainFrames, 2);
sigma = std(allTrainFrames, 0, 2);

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ (sigma + eps);
end
for i = 1:numel(XValidation)
    XValidation{i} = (XValidation{i} - mu) ./ (sigma + eps);
end

% --- 3. Define Hybrid CNN-BiLSTM Architecture ---
numFeatures = size(XTrain{1}, 1); % 60
numClasses = numel(unique(YTrain));

layers = [
    sequenceInputLayer(numFeatures, "Name", "input")
    
    % CNN part to find local feature patterns
    convolution1dLayer(5, 64, "Padding", "causal")
    reluLayer
    layerNormalizationLayer
    
    % BiLSTM part to learn temporal dependencies
    bilstmLayer(100, "OutputMode", "last")
    dropoutLayer(0.3)
    
    % Classifier part
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% --- 4. Set Training Options with Validation ---
options = trainingOptions("adam", ...
    "MaxEpochs", 45, ...
    "MiniBatchSize", 32, ...
    "Shuffle", "every-epoch", ...
    "Plots", "training-progress", ...
    "ValidationData", {XValidation, YValidation}, ...
    "ValidationFrequency", 30, ...
    "Verbose", false);

% --- 5. Train the network ---
net = trainNetwork(XTrain, YTrain, layers, options);
save("emotionNet_sequential.mat", "net", "mu", "sigma");
disp("Sequential Training Completed Successfully");
