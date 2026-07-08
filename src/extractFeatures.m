function feat = extractFeatures(x, fs0)
    x = preprocessAudio(x, fs0); % Requires custom implementation
    fs = 16000;
    
    % Define frame parameters
    windowLength = floor(0.025 * fs);
    overlapLength = floor(0.015 * fs);
    win = hamming(windowLength, 'periodic');

    % --- MFCC Features (as a sequence) ---
    [coeffs, delta, deltaDelta] = mfcc(x, fs, 'Window', win, 'OverlapLength', overlapLength, 'NumCoeffs', 13);
    featMFCC = [coeffs, delta, deltaDelta]'; % (39 x N_Frames)

    % --- Prosody Features (as a sequence) ---
    [f0, ~] = pitch(x, fs, 'WindowLength', windowLength, 'OverlapLength', overlapLength, 'Method', 'SRH');
    numFrames = size(featMFCC, 2);

    % Align f0 length to match MFCC frames
    if length(f0) > numFrames
        f0 = f0(1:numFrames);
    elseif length(f0) < numFrames
        f0_padded = zeros(numFrames, 1);
        f0_padded(1:length(f0)) = f0;
        f0 = f0_padded;
    end
    f0 = (f0 - mean(f0)) / (std(f0) + eps); % Normalize
    featProsody = f0'; % (1 x N_Frames)

    % --- HHT Features (Static, replicated) ---
    imf = emd(x, 'MaxNumIMF', 5);
    featHHT = [];
    for k = 1:size(imf, 2)
        z = hilbert(imf(:,k));
        A = abs(z);
        phase = unwrap(angle(z));
        F = [0; diff(phase)] * (fs/(2*pi));
        featHHT = [featHHT, mean(A), std(A), mean(F), std(F)];
    end
    
    % Replicate the 1x20 HHT vector to match N_Frames context
    featHHT_seq = repmat(featHHT', 1, numFrames); % (20 x N_Frames)

    % --- Combine all features ---
    feat = [featMFCC; featProsody; featHHT_seq]; % Total 60 features x N_Frames

    % --- Standardize Sequence Length ---
    fixedLength = 100;
    if size(feat, 2) > fixedLength
        feat = feat(:, 1:fixedLength);
    elseif size(feat, 2) < fixedLength
        feat(:, end+1:fixedLength) = 0; % Pad
    end
end
