#!/usr/bin/env python3
"""
Training Script for Guitar Technique Detection
Improved training with better synthetic data and real audio learning capabilities
"""

import argparse
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

from config.settings import Settings
from detection.feature_extractor import FeatureExtractor
from utils.logger import Logger


class TechniqueTrainer:
    """Enhanced trainer for guitar technique detection"""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = Logger()
        self.feature_extractor = FeatureExtractor(self.settings)
        
        self.sample_rate = self.settings.get('audio.sample_rate', 44100)
        self.window_size = self.settings.get('detection.window_size', 2048)
        
        # Training parameters
        self.techniques = ['none', 'chugging', 'harmonic']
        self.synthetic_samples_per_class = 200  # More samples for better learning
        
    def create_improved_synthetic_data(self) -> tuple:
        """Create more realistic synthetic training data"""
        self.logger.info("Creating improved synthetic training data...")
        
        features_list = []
        labels_list = []
        
        for label_idx, technique in enumerate(self.techniques):
            self.logger.info(f"Generating {self.synthetic_samples_per_class} samples for {technique}")
            
            for i in range(self.synthetic_samples_per_class):
                if technique == 'none':
                    audio = self._generate_silence_with_noise()
                elif technique == 'chugging':
                    audio = self._generate_realistic_chugging()
                elif technique == 'harmonic':
                    audio = self._generate_realistic_pinch_harmonic()
                
                # Extract features
                features = self.feature_extractor.extract_features(audio)
                if features is not None and len(features) > 0:
                    features_list.append(features)
                    labels_list.append(label_idx)
        
        self.logger.info(f"Generated {len(features_list)} synthetic training samples")
        return np.array(features_list), np.array(labels_list)
    
    def _generate_silence_with_noise(self) -> np.ndarray:
        """Generate background noise/silence"""
        # Very low amplitude noise
        audio = np.random.normal(0, 0.02, self.window_size)
        
        # Occasionally add very quiet single notes
        if np.random.random() < 0.3:
            freq = np.random.uniform(100, 400)
            t = np.linspace(0, self.window_size/self.sample_rate, self.window_size)
            note = 0.1 * np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-t * 2)
            audio += note * envelope
        
        return audio
    
    def _generate_realistic_chugging(self) -> np.ndarray:
        """Generate more realistic palm-muted chugging sounds"""
        t = np.linspace(0, self.window_size/self.sample_rate, self.window_size)
        
        # Low frequency fundamentals (low E, A, D strings)
        fundamental_freqs = [82.4, 110.0, 146.8]  # E, A, D
        base_freq = np.random.choice(fundamental_freqs)
        
        # Create palm-muted characteristic
        audio = np.zeros(self.window_size)
        
        # Sharp attack with quick decay (palm muting effect)
        attack_samples = int(0.01 * self.sample_rate)  # 10ms attack
        decay_samples = int(0.1 * self.sample_rate)   # 100ms decay
        
        # Create envelope for palm-muted attack
        envelope = np.zeros(self.window_size)
        if attack_samples < len(envelope):
            # Sharp attack
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            # Quick decay
            decay_end = min(attack_samples + decay_samples, len(envelope))
            decay_length = decay_end - attack_samples
            envelope[attack_samples:decay_end] = np.exp(-np.linspace(0, 4, decay_length))
        
        # Fundamental frequency with limited harmonics (palm muting dampens harmonics)
        audio += np.sin(2 * np.pi * base_freq * t) * 0.7
        audio += np.sin(2 * np.pi * base_freq * 2 * t) * 0.2  # Weak 2nd harmonic
        
        # Apply palm-muting envelope
        audio *= envelope
        
        # Add some string noise and fret buzz (characteristic of aggressive playing)
        noise = np.random.normal(0, 0.05, self.window_size)
        buzz_freq = base_freq * 0.5
        buzz = 0.1 * np.sin(2 * np.pi * buzz_freq * t) * np.random.random()
        audio += noise + buzz
        
        # Sometimes add rhythmic patterns (multiple chugs)
        if np.random.random() < 0.4:
            # Create a second chug
            second_chug_start = int(0.15 * self.sample_rate)
            if second_chug_start < len(audio) - decay_samples:
                second_envelope = np.zeros(self.window_size)
                second_end = min(second_chug_start + decay_samples, len(second_envelope))
                second_length = second_end - second_chug_start
                second_envelope[second_chug_start:second_end] = 0.8 * np.exp(-np.linspace(0, 3, second_length))
                
                second_audio = np.sin(2 * np.pi * base_freq * t) * 0.5
                audio += second_audio * second_envelope
        
        return audio
    
    def _generate_realistic_pinch_harmonic(self) -> np.ndarray:
        """Generate realistic pinch harmonic sounds"""
        t = np.linspace(0, self.window_size/self.sample_rate, self.window_size)
        
        # Pinch harmonics typically occur on specific fret positions
        # They emphasize specific harmonics (2nd, 3rd, 4th, 5th, 7th, 12th)
        base_freq = np.random.uniform(80, 200)  # Fundamental (often not heard much)
        
        # Common pinch harmonic nodes and their frequencies
        harmonic_nodes = [2, 3, 4, 5, 7, 12]  # Common pinch harmonic positions
        selected_harmonic = np.random.choice(harmonic_nodes)
        
        # Create the squealing harmonic sound
        audio = np.zeros(self.window_size)
        
        # The fundamental is usually weak or absent
        audio += 0.1 * np.sin(2 * np.pi * base_freq * t)
        
        # Emphasize the selected harmonic (this is what creates the squeal)
        harmonic_freq = base_freq * selected_harmonic
        audio += 0.8 * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Add related harmonics (harmonic series around the main one)
        audio += 0.3 * np.sin(2 * np.pi * harmonic_freq * 1.5 * t)  # Fifth above
        audio += 0.2 * np.sin(2 * np.pi * harmonic_freq * 2 * t)    # Octave above
        
        # Pinch harmonics often have a sharp attack but longer sustain
        attack_time = 0.02  # 20ms attack
        sustain_time = 0.4   # 400ms sustain
        
        attack_samples = int(attack_time * self.sample_rate)
        sustain_samples = int(sustain_time * self.sample_rate)
        
        envelope = np.zeros(self.window_size)
        
        # Sharp attack
        if attack_samples < len(envelope):
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Sustained decay (longer than chugging)
        sustain_end = min(attack_samples + sustain_samples, len(envelope))
        sustain_length = sustain_end - attack_samples
        if sustain_length > 0:
            envelope[attack_samples:sustain_end] = np.exp(-np.linspace(0, 2, sustain_length))
        
        # Apply envelope
        audio *= envelope
        
        # Add some high-frequency harmonics that make it "squealy"
        for i in range(2, 6):
            freq = harmonic_freq * i
            if freq < self.sample_rate / 2:  # Avoid aliasing
                amplitude = 0.1 / i  # Decreasing amplitude
                audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add some subtle vibrato (common in pinch harmonics)
        vibrato_freq = np.random.uniform(4, 8)  # 4-8 Hz vibrato
        vibrato_depth = 0.02
        vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)
        audio *= vibrato
        
        return audio
    
    def train_from_wav_files(self, data_dir: str) -> tuple:
        """Train from organized WAV files in folders"""
        self.logger.info(f"Loading training data from: {data_dir}")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            self.logger.error(f"Data directory not found: {data_dir}")
            return None, None
        
        features_list = []
        labels_list = []
        
        for technique_idx, technique in enumerate(self.techniques):
            technique_dir = data_path / technique
            if not technique_dir.exists():
                self.logger.warning(f"No directory found for technique: {technique}")
                continue
            
            wav_files = list(technique_dir.glob("*.wav")) + list(technique_dir.glob("*.WAV"))
            self.logger.info(f"Found {len(wav_files)} files for {technique}")
            
            for wav_file in wav_files:
                try:
                    # Load audio file
                    audio_data, _ = librosa.load(wav_file, sr=self.sample_rate, mono=True)
                    
                    # Process in overlapping windows
                    window_hop = self.window_size // 2
                    for i in range(0, len(audio_data) - self.window_size, window_hop):
                        window = audio_data[i:i + self.window_size]
                        
                        # Only use windows with sufficient energy
                        if np.sqrt(np.mean(window**2)) > 0.01:  # RMS threshold
                            features = self.feature_extractor.extract_features(window)
                            if features is not None:
                                features_list.append(features)
                                labels_list.append(technique_idx)
                
                except Exception as e:
                    self.logger.error(f"Error processing {wav_file}: {e}")
        
        if features_list:
            self.logger.info(f"Loaded {len(features_list)} samples from WAV files")
            return np.array(features_list), np.array(labels_list)
        else:
            self.logger.warning("No valid samples loaded from WAV files")
            return None, None
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train the technique detection model"""
        self.logger.info("Training technique detection model...")
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train model with optimized parameters
        model = RandomForestClassifier(
            n_estimators=200,      # More trees for better performance
            max_depth=15,          # Deeper trees
            min_samples_split=5,   # Prevent overfitting
            min_samples_leaf=3,    # Prevent overfitting
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        self.logger.info(f"Training accuracy: {train_score:.3f}")
        self.logger.info(f"Validation accuracy: {test_score:.3f}")
        
        # Detailed evaluation
        y_pred = model.predict(X_test_scaled)
        
        self.logger.info("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=self.techniques)
        self.logger.info(f"\n{report}")
        
        self.logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        self.logger.info(f"\n{cm}")
        
        # Feature importance
        feature_names = self.feature_extractor.get_feature_names()
        importances = model.feature_importances_
        
        self.logger.info("\nTop 10 Most Important Features:")
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            self.logger.info(f"{i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
        
        return model, scaler
    
    def save_model(self, model, scaler):
        """Save the trained model and scaler"""
        try:
            model_dir = Path('models')
            model_dir.mkdir(exist_ok=True)
            
            # Save model and scaler
            joblib.dump(model, model_dir / 'technique_model.pkl')
            joblib.dump(scaler, model_dir / 'technique_scaler.pkl')
            
            # Save training info
            training_info = {
                'techniques': self.techniques,
                'feature_count': self.feature_extractor.get_feature_count(),
                'sample_rate': self.sample_rate,
                'window_size': self.window_size,
                'model_type': 'RandomForestClassifier',
                'synthetic_samples_per_class': self.synthetic_samples_per_class
            }
            
            with open(model_dir / 'training_info.json', 'w') as f:
                json.dump(training_info, f, indent=2)
            
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def create_sample_dataset_structure(self, output_dir: str):
        """Create directory structure for organizing training samples"""
        output_path = Path(output_dir)
        
        for technique in self.techniques:
            technique_dir = output_path / technique
            technique_dir.mkdir(parents=True, exist_ok=True)
            
            # Create readme file with instructions
            readme_content = f"""# {technique.upper()} Samples

Place your {technique} guitar recordings (WAV files) in this directory.

Guidelines:
- Use clean, isolated recordings of {technique} technique
- WAV format, any sample rate (will be resampled to 44.1kHz)
- Each file should be 2-10 seconds long
- Name files descriptively (e.g., "{technique}_sample_001.wav")

The training script will automatically process all WAV files in this directory.
"""
            
            with open(technique_dir / 'README.md', 'w') as f:
                f.write(readme_content)
        
        self.logger.info(f"Created sample dataset structure in: {output_dir}")
        self.logger.info("Add your WAV files to the appropriate technique folders, then run:")
        self.logger.info(f"python train_technique_model.py --wav-data {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train guitar technique detection model")
    
    parser.add_argument('--synthetic-only', action='store_true',
                       help='Train only on improved synthetic data')
    parser.add_argument('--wav-data', type=str,
                       help='Directory with technique folders containing WAV files')
    parser.add_argument('--create-structure', type=str,
                       help='Create sample dataset directory structure')
    parser.add_argument('--combined', action='store_true',
                       help='Use both synthetic data and WAV files (if available)')
    
    args = parser.parse_args()
    
    if args.create_structure:
        trainer = TechniqueTrainer()
        trainer.create_sample_dataset_structure(args.create_structure)
        return
    
    # Initialize trainer
    trainer = TechniqueTrainer()
    
    # Collect training data
    all_features = []
    all_labels = []
    
    # Always create improved synthetic data
    if args.synthetic_only or args.combined or not args.wav_data:
        print("Creating improved synthetic training data...")
        synthetic_X, synthetic_y = trainer.create_improved_synthetic_data()
        all_features.append(synthetic_X)
        all_labels.append(synthetic_y)
    
    # Load WAV data if requested
    if args.wav_data and (args.combined or not args.synthetic_only):
        print(f"Loading WAV training data from: {args.wav_data}")
        wav_X, wav_y = trainer.train_from_wav_files(args.wav_data)
        if wav_X is not None:
            all_features.append(wav_X)
            all_labels.append(wav_y)
        else:
            print("No WAV data loaded, falling back to synthetic only")
    
    if not all_features:
        print("No training data available!")
        return 1
    
    # Combine all data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"Total training samples: {len(X)}")
    for i, technique in enumerate(trainer.techniques):
        count = np.sum(y == i)
        print(f"  {technique}: {count} samples")
    
    # Train model
    model, scaler = trainer.train_model(X, y)
    
    # Save model
    trainer.save_model(model, scaler)
    
    print("\nTraining complete! The improved model has been saved.")
    print("You can now test it with the real-time app or batch processing.")


if __name__ == "__main__":
    exit(main())