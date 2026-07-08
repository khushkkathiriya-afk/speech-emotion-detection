# Real-Time Emotion Detection using Signal Processing and Machine Learning

This repository contains the source code for a Speech Emotion Recognition (SER) system that utilizes a hybrid CNN-BiLSTM architecture. The project is designed to handle the non-stationary, non-linear nature of human speech by extracting complex emotional cues mapped across time.

## Methodology

### 1. Preprocessing
Raw audio is standardized to $f_s = 16000$ Hz, normalized to a range of $[-1, 1]$, and passed through a bandpass IIR filter (50 Hz to 4000 Hz). Silent frames are removed using energy-based Voice Activity Detection (VAD).

### 2. Feature Extraction
The system captures both perceptual and adaptive data over 1-second audio frames:
* **Perceptual (Sequence):** Mel-Frequency Cepstral Coefficients (MFCCs) and Prosody properties (fundamental frequency $f_0$ contour and signal energy).
* **Adaptive (Context):** The Hilbert-Huang Transform (HHT) applies Empirical Mode Decomposition (EMD) into Intrinsic Mode Functions (IMFs). Instantaneous frequency and amplitude are calculated to construct a highly accurate map of pitch dynamics.

### 3. Classification
A deep learning model analyzes temporal data flows.
* **1D-CNN Layers:** Scan for localized spatial patterns (e.g., rapid pitch changes combined with specific spectral shapes).
* **BiLSTM Layers:** Processes feature sequences bidirectionally to ascertain broader temporal dependencies.

## Future Scope
* **Edge AI:** Application of Post-Training Quantization (PTQ) to convert parameters to 8-bit integers, accelerating edge-device inference (e.g., Raspberry Pi 5).
* **Deep Unfolding (U-EMD):** Migrating the iterative EMD calculation into a neural network for single-pass extraction.
* **MIMO Acoustic Sensing:** Integrating Blind Source Separation (BSS) for emotion detection across multiple speakers.
