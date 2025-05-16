import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import soundfile as sf
from tqdm import tqdm

"""
Data Preprocessing as described in the paper:
1. Use EmoDB and RAVDESS datasets
2. Resample all audio to 22,025 Hz
3. Adjust all audio to 10 seconds by padding
4. Convert to mel-spectrograms using STFT (hop length 256, window size 1024)
5. Apply Z-score normalization
"""

# Paths to datasets (update these)
EMODB_PATH = "D:\\downloaaaad\\EmoDB\\wav"
RAVDESS_PATH = "D:\\downloaaaad\\RAVDESS"
OUTPUT_PATH = "D:\\downloaaaad\\output\\preprocessed_data"

# Create output directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "mel_specs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "processed_audio"), exist_ok=True)

# Parameters as specified in the paper
TARGET_SR = 22025  # 22,025 Hz sampling rate
TARGET_LENGTH = TARGET_SR * 10  # 10 seconds duration
HOP_LENGTH = 256  # As mentioned in paper
WINDOW_SIZE = 1024  # As mentioned in paper
N_MELS = 80  # Standard value for mel spectrogram

# EmoDB mapping of emotion codes to labels
EMODB_EMOTION_MAP = {
    'W': 'anger',      # Wut/Ärger
    'L': 'boredom',    # Not used in the experiment
    'E': 'disgust',    # Ekel
    'A': 'fear',       # Angst
    'F': 'happiness',  # Freude
    'T': 'sadness',    # Trauer
    'N': 'neutral'     # Neutral
}

# RAVDESS mapping of emotion codes to labels
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '03': 'happiness',
    '04': 'sadness',
    '05': 'anger',
    '06': 'fear',
    '07': 'disgust'
}

# Emotions used in the paper based on Table 1
USED_EMOTIONS = ['neutral', 'anger', 'sadness', 'fear', 'happiness', 'disgust']

def process_audio(audio_path, target_sr=TARGET_SR, target_length=TARGET_LENGTH):
    """
    Process an audio file according to paper specifications:
    - Resample to 22,025 Hz
    - Adjust length to 10 seconds
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=target_sr)
    
    # Adjust length to 10 seconds
    if len(y) < target_length:
        # Pad shorter samples
        y = np.pad(y, (0, target_length - len(y)), 'constant')
    else:
        # Trim longer samples
        y = y[:target_length]
    
    return y

def create_mel_spectrogram(y, sr=TARGET_SR, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Create mel-spectrogram using STFT with specified parameters.
    """
    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def normalize_mel_spectrogram(mel_spec_db):
    """
    Apply Z-score normalization to mel-spectrogram as mentioned in the paper.
    """
    # Reshape for standardization
    mel_shape = mel_spec_db.shape
    mel_spec_flat = mel_spec_db.reshape((mel_spec_db.shape[0], -1))
    
    # Apply Z-score normalization
    scaler = StandardScaler()
    mel_spec_normalized = scaler.fit_transform(mel_spec_flat)
    
    # Reshape back to original shape
    mel_spec_normalized = mel_spec_normalized.reshape(mel_shape)
    
    return mel_spec_normalized

def process_emodb():
    """
    Process EmoDB dataset according to the paper.
    """
    print("Processing EmoDB dataset...")
    
    metadata = []
    
    for filename in tqdm(os.listdir(EMODB_PATH)):
        if not filename.endswith('.wav'):
            continue
            
        # Extract emotion code (e.g., 03a01Fa.wav -> 'F' is emotion code)
        emotion_code = filename[5]
        if emotion_code not in EMODB_EMOTION_MAP:
            continue
            
        emotion = EMODB_EMOTION_MAP[emotion_code]
        
        # Skip emotions not used in the paper
        if emotion not in USED_EMOTIONS:
            continue
            
        file_path = os.path.join(EMODB_PATH, filename)
        
        # Process audio
        y = process_audio(file_path)
        
        # Create output directories
        emotion_dir = os.path.join(OUTPUT_PATH, "processed_audio", "emodb", emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        # Save processed audio
        output_audio_path = os.path.join(emotion_dir, filename)
        sf.write(output_audio_path, y, TARGET_SR)
        
        # Create mel-spectrogram
        mel_spec_db = create_mel_spectrogram(y)
        
        # Normalize mel-spectrogram
        mel_spec_normalized = normalize_mel_spectrogram(mel_spec_db)
        
        # Save mel-spectrogram
        mel_dir = os.path.join(OUTPUT_PATH, "mel_specs", "emodb", emotion)
        os.makedirs(mel_dir, exist_ok=True)
        
        mel_path = os.path.join(mel_dir, f"{os.path.splitext(filename)[0]}.npy")
        np.save(mel_path, mel_spec_normalized)
        
        metadata.append({
            'dataset': 'emodb',
            'filename': filename,
            'emotion': emotion,
            'audio_path': output_audio_path,
            'mel_spec_path': mel_path
        })
    
    return pd.DataFrame(metadata)

def process_ravdess():
    """
    Process RAVDESS dataset according to the paper.
    """
    print("Processing RAVDESS dataset...")
    
    metadata = []
    
    # Process all actor directories
    for actor_dir in tqdm(os.listdir(RAVDESS_PATH)):
        if not actor_dir.startswith('Actor_'):
            continue
            
        actor_path = os.path.join(RAVDESS_PATH, actor_dir)
        
        for filename in os.listdir(actor_path):
            if not filename.endswith('.wav'):
                continue
                
            # Extract emotion code (format: 03-01-01-01-01-01-01.wav)
            parts = filename.split('-')
            if len(parts) < 3:
                continue
                
            emotion_code = parts[2]
            if emotion_code not in RAVDESS_EMOTION_MAP:
                continue
                
            emotion = RAVDESS_EMOTION_MAP[emotion_code]
            
            # Skip emotions not used in the paper
            if emotion not in USED_EMOTIONS:
                continue
                
            file_path = os.path.join(actor_path, filename)
            
            # Process audio
            y = process_audio(file_path)
            
            # Create output directories
            emotion_dir = os.path.join(OUTPUT_PATH, "processed_audio", "ravdess", emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            # Save processed audio
            output_audio_path = os.path.join(emotion_dir, filename)
            sf.write(output_audio_path, y, TARGET_SR)
            
            # Create mel-spectrogram
            mel_spec_db = create_mel_spectrogram(y)
            
            # Normalize mel-spectrogram
            mel_spec_normalized = normalize_mel_spectrogram(mel_spec_db)
            
            # Save mel-spectrogram
            mel_dir = os.path.join(OUTPUT_PATH, "mel_specs", "ravdess", emotion)
            os.makedirs(mel_dir, exist_ok=True)
            
            mel_path = os.path.join(mel_dir, f"{os.path.splitext(filename)[0]}.npy")
            np.save(mel_path, mel_spec_normalized)
            
            metadata.append({
                'dataset': 'ravdess',
                'filename': filename,
                'emotion': emotion,
                'audio_path': output_audio_path,
                'mel_spec_path': mel_path
            })
    
    return pd.DataFrame(metadata)

def visualize_sample_spectrograms(metadata_df, num_samples=2):
    """
    Visualize sample mel-spectrograms from each emotion category.
    This is similar to Figure 3 in the paper.
    """
    # Group by dataset and emotion
    for dataset in ['emodb', 'ravdess']:
        plt.figure(figsize=(15, 10))
        
        dataset_df = metadata_df[metadata_df['dataset'] == dataset]
        
        for i, emotion in enumerate(USED_EMOTIONS):
            emotion_df = dataset_df[dataset_df['emotion'] == emotion]
            
            if emotion_df.empty:
                continue
                
            # Get samples
            samples = emotion_df.sample(min(num_samples, len(emotion_df)))
            
            for j, (_, row) in enumerate(samples.iterrows()):
                mel_path = row['mel_spec_path']
                mel_spec = np.load(mel_path)
                
                plt.subplot(len(USED_EMOTIONS), num_samples, i * num_samples + j + 1)
                librosa.display.specshow(
                    mel_spec,
                    sr=TARGET_SR,
                    hop_length=HOP_LENGTH,
                    x_axis='time',
                    y_axis='mel'
                )
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"{dataset} - {emotion}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f"{dataset}_mel_spectrograms.png"))
        plt.close()

def main():
    """
    Main function to process both datasets.
    """
    # Process EmoDB dataset
    emodb_metadata = process_emodb()
    
    # Process RAVDESS dataset
    ravdess_metadata = process_ravdess()
    
    # Combine metadata
    all_metadata = pd.concat([emodb_metadata, ravdess_metadata], ignore_index=True)
    
    # Save metadata
    all_metadata.to_csv(os.path.join(OUTPUT_PATH, "metadata.csv"), index=False)
    
    # Print dataset statistics (Table 1 in the paper)
    print("\nDataset Statistics:")
    emotion_counts = all_metadata.groupby(['dataset', 'emotion']).size().unstack(fill_value=0)
    print(emotion_counts)
    
    # Visualize sample spectrograms
    visualize_sample_spectrograms(all_metadata)
    
    print(f"\nTotal processed files: {len(all_metadata)}")
    print(f"EmoDB files: {len(emodb_metadata)}")
    print(f"RAVDESS files: {len(ravdess_metadata)}")

if __name__ == "__main__":
    main()