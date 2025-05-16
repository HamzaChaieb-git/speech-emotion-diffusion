# 🎙️ Speech Emotion Recognition with Diffusion Modeling

![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio_Processing-76B900.svg?style=flat)
![Hugging Face](https://img.shields.io/badge/🤗_Transformers-yellow.svg?style=flat)
![Diffusion](https://img.shields.io/badge/Diffusion_Models-6A5ACD.svg?style=flat)
![Mel-Spectrogram](https://img.shields.io/badge/Mel--Spectrogram-8A2BE2.svg?style=flat)
![EmoDB](https://img.shields.io/badge/EmoDB-Dataset-FF6F61.svg?style=flat)
![RAVDESS](https://img.shields.io/badge/RAVDESS-Dataset-1E90FF.svg?style=flat)

A comprehensive implementation of the paper "A Generation of Enhanced Data by Variational Autoencoders and Diffusion Modeling", featuring advanced diffusion-based speech emotion enhancement, improved recognition accuracy, and state-of-the-art speech processing techniques.

## 📝 Overview

This project implements an end-to-end system for enhancing speech emotion clarity through diffusion models with the following components:

- **Data Preprocessing Pipeline**: Advanced audio processing, feature extraction, and normalization
- **Diffusion Modeling**: State-of-the-art techniques to enhance emotional salience in speech
- **Emotion Recognition**: ResNet-based classifier with attention mechanisms
- **Mel-Spectrogram Engineering**: Specialized audio feature extraction for emotion analysis
- **Hugging Face Integration**: Evaluation with pre-trained transformer models
- **Visualization System**: Comprehensive analysis and comparison tools

## 🚀 Features

### Advanced Preprocessing

- Resampling from 16/48 kHz to unified 22.025 kHz sampling rate
- Intelligent audio padding to standardized 10-second length
- Mel-spectrogram generation with optimized parameters (hop length 256, window size 1024)
- Z-score normalization for feature standardization

### Diffusion-Based Enhancement

- Noise injection and extraction techniques for emotional clarity
- Emotion-specific embedding mechanisms
- Utterance style preservation during enhancement
- Conditional generation based on emotion targets

### Model Architecture

- ResNet-based backbone with attention mechanisms
- Emotion-specific classifier heads
- Specialized features for distinct emotional patterns
- Multi-stage training with mixup and gradient accumulation

### Evaluation Framework

- Comprehensive metrics for emotion recognition assessment
- Cross-dataset validation methodology
- Comparative analysis with state-of-the-art models
- Hugging Face transformer model integration

## 📊 Datasets

### 🎭 EmoDB (Berlin Database of Emotional Speech)
- **Languages**: German
- **Emotions**: Neutral, Happiness, Sadness, Anger, Fear, Disgust
- **Speakers**: 10 (5 male, 5 female)
- **Original Sampling Rate**: 16 kHz
- **Size**: 454 utterances

### 🎵 RAVDESS (Ryerson Audio-Visual Database)
- **Languages**: English
- **Emotions**: Neutral, Happiness, Sadness, Anger, Fear, Disgust (+ others not used)
- **Speakers**: 24 (12 male, 12 female)
- **Original Sampling Rate**: 48 kHz
- **Size**: 1,056 utterances

## 📁 Repository Structure

```
speech-emotion-diffusion/
├── EmoDB/wav/                 # EmoDB dataset files
├── RAVDESS/                   # RAVDESS dataset files
├── preprocessed_data/         # Processed audio and mel-spectrograms
├── spectrogram_comparisons/   # Visual comparisons of spectrograms
├── _pycache_/                 # Compiled Python files
├── data prep.py               # Data preprocessing pipeline
├── enhanced_fine.py           # Fine-tuning implementation for SER
└── HF.py                      # Hugging Face model evaluation
```

## ⚙️ Usage

### 1. Data Preprocessing

```bash
python data\ prep.py
```

This script processes both EmoDB and RAVDESS datasets through:
- Audio standardization (length, sampling rate)
- Mel-spectrogram generation
- Feature normalization
- Dataset organization by emotion

### 2. Model Training and Fine-tuning

```bash
python enhanced_fine.py
```

Implements the core diffusion model and emotion recognition system:
- ResNet-based emotion classifier
- Attention mechanism for emotion focus
- Advanced training techniques
- Performance monitoring and visualization

### 3. Evaluation with Hugging Face Models

```bash
python HF.py
```

Evaluates the enhanced data quality through:
- Multiple pre-trained transformer models
- Comprehensive performance metrics
- Visualization generation
- Comparative analysis

## 📚 References

1. Kim, Y.-J.; Lee, S.-P. A Generation of Enhanced Data by Variational Autoencoders and Diffusion Modeling. Electronics 2024, 13, 1314. https://doi.org/10.3390/electronics13071314
2. Livingstone, S.R.; Russo, F.A. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 2018, 13, e0196391.
3. Burkhardt, F.; Paeschke, A.; Rolfes, M.; Sendlmeier, W.; Weiss, B. A Database of German Emotional Speech. In Proceedings of Interspeech, Lisbon, Portugal, 2005.
