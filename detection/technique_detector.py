"""
Technique Detector - AI-driven detection of guitar playing techniques
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Optional

from detection.feature_extractor import FeatureExtractor
from utils.logger import Logger

class TechniqueDetector:
    """Detects guitar playing techniques using machine learning"""
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = Logger()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(settings)
        
        # ML model and scaler
        self.model = None
        self.scaler = None
        
        # Technique labels
        self.technique_labels = ['none', 'chugging', 'harmonic']
        
        # Detection parameters
        self.detection_window_size = settings.get('detection.window_size', 2048)
        self.confidence_threshold = settings.get('detection.confidence_threshold', 0.7)
        
        # State tracking
        self.last_detection = None
        self.detection_history = []
        self.history_size = 5
        
        # Initialize or load model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the ML model"""
        model_path = self.settings.get('detection.model_path', 'models/technique_model.pkl')
        scaler_path = self.settings.get('detection.scaler_path', 'models/technique_scaler.pkl')
        
        # Try to load existing model
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.logger.info("Loaded pre-trained technique detection model")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load model: {e}")
        
        # Create new model if loading failed
        self._create_default_model()
    
    def _create_default_model(self):
        """Create a default model with basic training data"""
        self.logger.info("Creating default technique detection model")
        
        # Create model and scaler
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Generate synthetic training data for initialization
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate synthetic training data for model initialization"""
        try:
            # Create synthetic audio samples for each technique
            sample_rate = self.settings.get('audio.sample_rate', 44100)
            samples_per_class = 50
            
            features_list = []
            labels_list = []
            
            # Generate features for each technique
            for label_idx, technique in enumerate(self.technique_labels):
                for _ in range(samples_per_class):
                    # Generate synthetic audio based on technique characteristics
                    if technique == 'none':
                        # Low amplitude, minimal harmonics
                        audio = np.random.normal(0, 0.1, self.detection_window_size)
                    elif technique == 'chugging':
                        # High amplitude, low frequency emphasis, rhythmic
                        freq = np.random.uniform(80, 200)  # Low frequency range
                        t = np.linspace(0, self.detection_window_size/sample_rate, self.detection_window_size)
                        audio = np.sin(2 * np.pi * freq * t) * 0.8
                        # Add some noise and harmonics
                        audio += np.random.normal(0, 0.2, self.detection_window_size)
                        audio += 0.3 * np.sin(4 * np.pi * freq * t)  # Second harmonic
                    elif technique == 'harmonic':
                        # High frequency content, specific harmonic patterns
                        fundamental = np.random.uniform(200, 800)
                        t = np.linspace(0, self.detection_window_size/sample_rate, self.detection_window_size)
                        # Create harmonic series with emphasis on higher harmonics
                        audio = np.zeros(self.detection_window_size)
                        for harmonic in range(1, 8):
                            amplitude = 0.5 / harmonic  # Decreasing amplitude
                            if harmonic > 3:  # Emphasize higher harmonics for pinch harmonics
                                amplitude *= 3
                            audio += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)
                        audio *= 0.6
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(audio)
                    features_list.append(features)
                    labels_list.append(label_idx)
            
            # Train model
            X = np.array(features_list)
            y = np.array(labels_list)
            
            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            self.logger.info(f"Model trained with {len(X)} synthetic samples")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            self.logger.error(f"Failed to generate training data: {e}")
            # Create minimal fallback model
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a minimal fallback model"""
        self.logger.warning("Creating fallback model")
        
        # Create very simple training data
        X = np.random.random((30, 13))  # 13 features (typical for audio)
        y = np.random.randint(0, 3, 30)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def detect_technique(self, audio_data: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Detect playing technique from audio data
        
        Args:
            audio_data: Audio samples for analysis
            
        Returns:
            Tuple of (technique_name, confidence)
        """
        try:
            if self.model is None or self.scaler is None:
                return None, 0.0
            
            # Extract features from audio
            features = self.feature_extractor.extract_features(audio_data)
            
            if features is None or len(features) == 0:
                return None, 0.0
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict technique
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Get technique name
            technique = self.technique_labels[predicted_class] if predicted_class < len(self.technique_labels) else None
            
            # Apply smoothing with detection history
            technique, confidence = self._apply_temporal_smoothing(technique, confidence)
            
            # Update detection history
            self._update_detection_history(technique, confidence)
            
            return technique, confidence
            
        except Exception as e:
            self.logger.error(f"Technique detection error: {e}")
            return None, 0.0
    
    def _apply_temporal_smoothing(self, technique: str, confidence: float) -> Tuple[str, float]:
        """Apply temporal smoothing to reduce detection jitter"""
        if not self.detection_history:
            return technique, confidence
        
        # Count recent detections of the same technique
        recent_same = sum(1 for det in self.detection_history[-3:] if det[0] == technique)
        
        # Boost confidence if technique is consistent
        if recent_same >= 2:
            confidence = min(1.0, confidence * 1.2)
        
        # Reduce confidence for single detections
        elif recent_same == 0 and len(self.detection_history) >= 2:
            confidence *= 0.8
        
        return technique, confidence
    
    def _update_detection_history(self, technique: str, confidence: float):
        """Update detection history for temporal smoothing"""
        self.detection_history.append((technique, confidence))
        
        # Keep only recent history
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
    
    def _save_model(self):
        """Save the trained model and scaler"""
        try:
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, 'technique_model.pkl')
            scaler_path = os.path.join(model_dir, 'technique_scaler.pkl')
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def retrain_model(self, audio_samples: list, labels: list):
        """Retrain the model with new data"""
        try:
            if not audio_samples or not labels:
                return False
            
            # Extract features from all samples
            features_list = []
            for audio in audio_samples:
                features = self.feature_extractor.extract_features(audio)
                if features is not None:
                    features_list.append(features)
            
            if not features_list:
                return False
            
            # Convert labels to indices
            label_indices = []
            for label in labels:
                if label in self.technique_labels:
                    label_indices.append(self.technique_labels.index(label))
                else:
                    label_indices.append(0)  # Default to 'none'
            
            # Prepare training data
            X = np.array(features_list)
            y = np.array(label_indices)
            
            # Retrain scaler and model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            # Save updated model
            self._save_model()
            
            self.logger.info(f"Model retrained with {len(X)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            return False
    
    def get_supported_techniques(self) -> list:
        """Get list of supported techniques"""
        return self.technique_labels.copy()
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            'model_type': type(self.model).__name__ if self.model else None,
            'techniques': self.technique_labels,
            'confidence_threshold': self.confidence_threshold,
            'window_size': self.detection_window_size,
            'history_size': self.history_size
        }
