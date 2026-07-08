clear; clc;
fs = 16000;
emotions = ["Angry", "Happy", "Sad", "Neutral", "Fear", "Disgust", "Surprise"];

features = {}; % Must be a cell array
labels = {};   % Must be a cell array

for em = emotions
    folder = fullfile("Dataset", em);
    files = dir(fullfile(folder, "*.wav"));
    fprintf("Processing %s...\n", em);
    
    for i = 1:length(files)
        [x, fs0] = audioread(fullfile(files(i).folder, files(i).name));
        feat = extractFeatures(x, fs0); % Gets (60x100) matrix
        
        if ~isempty(feat)
            features{end+1} = feat;
            labels{end+1} = em;
        end
    end
end

labels = categorical(labels');
save("emotionFeatures_sequential.mat", "features", "labels");
