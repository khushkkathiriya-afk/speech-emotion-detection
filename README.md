# Real-Time Emotion Detection using Signal Processing and Machine Learning

This repository contains the source code for a Speech Emotion Recognition (SER) system that utilizes a hybrid CNN-BiLSTM architecture[cite: 1]. The project addresses a fundamental challenge in SER: the non-stationary, non-linear nature of human speech, where emotional information is encoded in rapid, time-varying modulations of pitch and energy[cite: 1].

## 📊 Dataset Access (Important)

Due to GitHub's file size constraints and best practices for data management, the raw audio dataset (`.wav` files) used to train this model is not hosted directly in this repository. 

**To run this project locally:**
1. Download the full dataset archive from here: https://drive.google.com/file/d/1dFyiT1KZPs3Z7mJeTAs29NEDad_FAhZT/view?usp=sharing
2. Extract the `.zip` file.
3. Place the extracted `dataset` folder directly into the root directory of this repository. The script `src/buildDataset.m` will automatically search for it.

---

## ⚙️ System Architecture & Methodology

The system is designed as a continuous pipeline flowing from raw audio input to a final emotion label[cite: 1].

### 1. Signal Preprocessing
To clean and standardize the raw audio, the following steps are applied[cite: 1]:
* **Resampling & Normalization:** All incoming audio is standardized to a sampling rate of $f_{s} = 16000$ Hz and normalized to an amplitude range of $[-1,1]$[cite: 1].
* **Bandpass Filtering:** An Infinite Impulse Response (IIR) filter is applied from 50 Hz to 4000 Hz to isolate vocal frequencies[cite: 1]. This LTI system is governed by the difference equation[cite: 1]: 
  $$y[n]=\sum_{k=0}^{M}b_{k}x[n-k]-\sum_{k=1}^{N}a_{k}y[n-k]$$
* **Silence Removal:** An energy-based Voice Activity Detection (VAD) mechanism removes frames where $|x[n]|<0.02$[cite: 1].
* **Standardization:** Signals are padded or truncated to exactly 1 second (16,000 samples)[cite: 1].

### 2. Hybrid Feature Extraction
The core of this system is the combination of two distinct feature extraction philosophies to create a high-resolution map of speech dynamics[cite: 1].

**Perceptual Features (Standard)**[cite: 1]:
* **MFCCs:** Mimic the human auditory system to capture the "timbre" or vocal tract shape[cite: 1]. 
* **Prosody:** Captures the "musical" components, tracking the fundamental frequency ($f_{0}$) contour (pitch) and signal energy (loudness)[cite: 1].

**Adaptive Features (Advanced)**[cite: 1]:
To handle transient emotional cues, the **Hilbert-Huang Transform (HHT)** is employed[cite: 1].
* **Empirical Mode Decomposition (EMD):** Adaptively sifts the signal into natural oscillatory components called Intrinsic Mode Functions (IMFs)[cite: 1]. 
* **Hilbert Transform:** Applied to each IMF to create an analytic signal $z_{i}(t)$, revealing exact, moment-by-moment energy and frequency modulations[cite: 1]:
  $$z_{i}(t)=c_{i}(t)+j\mathcal{H}\{c_{i}(t)\}=A_{i}(t)e^{j\phi_{i}(t)}$$
  From this, two critical features are extracted[cite: 1]:
  * **Instantaneous Amplitude (Energy):** $A_{i}(t)=\sqrt{c_{i}(t)^{2}+(\mathcal{H}\{c_{i}(t)\})^{2}}$[cite: 1].
  * **Instantaneous Frequency (Pitch):** $f_{i}(t)=\frac{1}{2\pi}\frac{d\phi_{i}(t)}{dt}$[cite: 1].

### 3. Classification (CNN-BiLSTM)
Extracted feature sequences are fed into a hybrid deep learning model[cite: 1].
* **1D-CNN Layers:** Act as spatial feature extractors, learning localized, complex patterns across the frequency and amplitude data[cite: 1].
* **BiLSTM Layers:** Process the CNN outputs bidirectionally over time to understand the contextual flow of the emotion[cite: 1]. The internal cell state flow is regulated mathematically via[cite: 1]:
  $$C_{t}=f_{t}\odot C_{t-1}+i_{t}\odot\tilde{C_{t}}$$

---

## 📈 Iterative Development & Model Correction

This project evolved through an iterative, methodologically sound process to ensure robustness[cite: 1].

* **Phase 1 (Static Analysis):** Initial attempts collapsed time-series data into static 1D vectors using mean and standard deviation statistics[cite: 1]. While this achieved $\approx 95\%$ training accuracy, it suffered from severe overfitting (lacking validation sets) and a feature-model mismatch, as BiLSTM networks require sequential inputs to function optimally[cite: 1].
* **Phase 2 (Sequential Framework):** The system was completely re-engineered[cite: 1]. Feature extraction was modified to output 2D matrices across time frames[cite: 1]. Training scripts were updated to handle manual data partitioning (80/20 splits) and perform rigorous validation[cite: 1]. This resolved previous defects, proving the model's ability to generalize to unseen speech[cite: 1].

---

## 🚀 Future Scope

The established framework paves the way for advanced deployments[cite: 1]:
1. **Edge AI:** Utilizing Post-Training Quantization (PTQ) to convert the 32-bit floating-point model to 8-bit integers, shrinking its footprint for real-time edge devices like the Raspberry Pi 5[cite: 1].
2. **Deep Unfolding for HHT (U-EMD):** Replacing the computationally heavy, iterative EMD sifting process with a "Deep Unfolded" neural network for single, fast forward-pass executions[cite: 1].
3. **MIMO Acoustic Sensing:** Implementing Blind Source Separation (BSS) via microphone arrays to analyze emotions from multiple speakers simultaneously[cite: 1].
