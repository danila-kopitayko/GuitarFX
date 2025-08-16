"""
Feature Extractor - Extracts audio features for technique detection
"""

import numpy as np
import librosa
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Optional, List

from utils.logger import Logger

class FeatureExtractor:
    """Extracts relevant audio features for guitar technique detection"""
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = Logger()
        
        # Audio parameters
        self.sample_rate = settings.get('audio.sample_rate', 44100)
        
        # Feature extraction parameters
        self.n_mfcc = settings.get('features.n_mfcc', 13)
        self.n_fft = settings.get('features.n_fft', 2048)
        self.hop_length = settings.get('features.hop_length', 512)
        
        # Frequency ranges for different techniques
        self.low_freq_range = (80, 300)    # Chugging range
        self.mid_freq_range = (300, 2000)  # General playing
        self.high_freq_range = (2000, 8000) # Harmonics range
        
    def extract_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract comprehensive features from audio data
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            Feature vector as numpy array, or None if extraction fails
        """
        try:
            if len(audio_data) == 0:
                return None
            
            # Ensure audio is normalized
            audio_data = self._normalize_audio(audio_data)
            
            features = []
            
            # Time domain features
            time_features = self._extract_time_domain_features(audio_data)
            features.extend(time_features)
            
            # Frequency domain features
            freq_features = self._extract_frequency_domain_features(audio_data)
            features.extend(freq_features)
            
            # Spectral features
            spectral_features = self._extract_spectral_features(audio_data)
            features.extend(spectral_features)
            
            # Harmonic features
            harmonic_features = self._extract_harmonic_features(audio_data)
            features.extend(harmonic_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return None
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping and ensure consistent scaling"""
        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        return audio_data
    
    def _extract_time_domain_features(self, audio_data: np.ndarray) -> List[float]:
        """Extract time domain features"""
        features = []
        
        # RMS energy
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        features.append(rms_energy)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        zcr = zero_crossings / len(audio_data)
        features.append(zcr)
        
        # Statistical moments
        features.append(np.mean(np.abs(audio_data)))  # Mean absolute value
        features.append(np.std(audio_data))           # Standard deviation
        features.append(skew(audio_data))             # Skewness
        features.append(kurtosis(audio_data))         # Kurtosis
        
        # Peak characteristics
        peaks, _ = signal.find_peaks(np.abs(audio_data), height=0.1)
        features.append(len(peaks) / len(audio_data))  # Peak density
        
        return features
    
    def _extract_frequency_domain_features(self, audio_data: np.ndarray) -> List[float]:
        """Extract frequency domain features"""
        features = []
        
        # Compute FFT
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)[:len(fft)//2]
        
        # Normalize magnitude spectrum
        if np.sum(magnitude) > 0:
            magnitude = magnitude / np.sum(magnitude)
        
        # Spectral centroid
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0
        features.append(spectral_centroid)
        
        # Spectral rolloff (95% of energy)
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            spectral_rolloff = 0
        features.append(spectral_rolloff)
        
        # Spectral flux (measure of spectral change)
        spectral_flux = np.sum(np.diff(magnitude) ** 2)
        features.append(spectral_flux)
        
        # Energy in frequency bands
        low_energy = self._get_band_energy(magnitude, freqs, self.low_freq_range)
        mid_energy = self._get_band_energy(magnitude, freqs, self.mid_freq_range)
        high_energy = self._get_band_energy(magnitude, freqs, self.high_freq_range)
        
        features.extend([low_energy, mid_energy, high_energy])
        
        # Band energy ratios
        total_energy = low_energy + mid_energy + high_energy
        if total_energy > 0:
            features.append(low_energy / total_energy)   # Low frequency ratio
            features.append(high_energy / total_energy)  # High frequency ratio
        else:
            features.extend([0, 0])
        
        return features
    
    def _extract_spectral_features(self, audio_data: np.ndarray) -> List[float]:
        """Extract spectral features using librosa"""
        features = []
        
        try:
            # MFCC features (first few coefficients)
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=5,  # Use fewer MFCCs for efficiency
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Take mean of MFCCs across time
            mfcc_mean = np.mean(mfcc, axis=1)
            features.extend(mfcc_mean)
            
        except Exception as e:
            self.logger.warning(f"MFCC extraction failed: {e}")
            features.extend([0] * 5)  # Fallback zeros
        
        try:
            # Chroma features (for harmonic content)
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Take variance of chroma (indicates harmonic stability)
            chroma_var = np.var(chroma, axis=1)
            features.append(np.mean(chroma_var))
            
        except Exception as e:
            self.logger.warning(f"Chroma extraction failed: {e}")
            features.append(0)
        
        return features
    
    def _extract_harmonic_features(self, audio_data: np.ndarray) -> List[float]:
        """Extract features specific to harmonic content detection"""
        features = []
        
        try:
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio_data)
            
            # Harmonic to percussive ratio
            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            
            if percussive_energy > 0:
                hp_ratio = harmonic_energy / percussive_energy
            else:
                hp_ratio = harmonic_energy
            
            features.append(hp_ratio)
            
            # Pitch stability (for detecting pinch harmonics)
            try:
                pitches, magnitudes = librosa.piptrack(
                    y=audio_data,
                    sr=self.sample_rate,
                    threshold=0.1
                )
                
                # Find the most prominent pitch
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    pitch_stability = 1.0 / (1.0 + np.std(pitch_values))
                    mean_pitch = np.mean(pitch_values)
                else:
                    pitch_stability = 0
                    mean_pitch = 0
                
                features.extend([pitch_stability, mean_pitch])
                
            except Exception as e:
                self.logger.warning(f"Pitch tracking failed: {e}")
                features.extend([0, 0])
        
        except Exception as e:
            self.logger.warning(f"Harmonic feature extraction failed: {e}")
            features.extend([0, 0, 0])
        
        return features
    
    def _get_band_energy(self, magnitude: np.ndarray, freqs: np.ndarray, freq_range: tuple) -> float:
        """Get energy in a specific frequency band"""
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        band_energy = np.sum(magnitude[mask])
        return band_energy
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features for debugging"""
        return [
            'rms_energy', 'zero_crossing_rate', 'mean_abs', 'std_dev', 'skewness', 'kurtosis', 'peak_density',
            'spectral_centroid', 'spectral_rolloff', 'spectral_flux',
            'low_freq_energy', 'mid_freq_energy', 'high_freq_energy',
            'low_freq_ratio', 'high_freq_ratio',
            'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
            'chroma_variance',
            'harmonic_percussive_ratio', 'pitch_stability', 'mean_pitch'
        ]
    
    def get_feature_count(self) -> int:
        """Get the expected number of features"""
        return len(self.get_feature_names())
