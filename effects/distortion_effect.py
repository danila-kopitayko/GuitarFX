"""
Distortion Effect - Cyberpunk-style digital distortion for chugging
"""

import numpy as np
from scipy import signal
from typing import Dict, Any

from effects.base_effect import BaseEffect

class DistortionEffect(BaseEffect):
    """Cyberpunk-style digital distortion effect for aggressive guitar chugging"""
    
    def _initialize_parameters(self, **kwargs):
        """Initialize distortion parameters"""
        self.parameters = {
            'drive': kwargs.get('drive', 0.7),        # Distortion amount (0.0 - 1.0)
            'tone': kwargs.get('tone', 0.6),          # Tone control (0.0 - 1.0)
            'output_gain': kwargs.get('output_gain', 0.8),  # Output level (0.0 - 1.0)
            'bit_crush': kwargs.get('bit_crush', 0.3),      # Digital bit crushing (0.0 - 1.0)
            'saturation': kwargs.get('saturation', 0.5),    # Analog-style saturation (0.0 - 1.0)
            'low_cut': kwargs.get('low_cut', 80),           # High-pass filter frequency (Hz)
            'high_cut': kwargs.get('high_cut', 8000),       # Low-pass filter frequency (Hz)
        }
    
    def _initialize_state(self):
        """Initialize distortion state"""
        self.state = {
            'input_history': np.zeros(4),     # For filtering
            'output_history': np.zeros(4),    # For filtering
            'saturation_state': 0.0,          # For saturation memory
        }
        
        # Create filter coefficients
        self._update_filters()
    
    def _update_filters(self):
        """Update filter coefficients based on tone parameters"""
        nyquist = self.sample_rate / 2
        
        # High-pass filter for low cut
        low_cut_norm = self.parameters['low_cut'] / nyquist
        low_cut_norm = np.clip(low_cut_norm, 0.001, 0.99)
        self.hp_b, self.hp_a = signal.butter(2, low_cut_norm, 'highpass')
        
        # Low-pass filter for high cut
        high_cut_norm = self.parameters['high_cut'] / nyquist
        high_cut_norm = np.clip(high_cut_norm, 0.001, 0.99)
        self.lp_b, self.lp_a = signal.butter(2, high_cut_norm, 'lowpass')
        
        # Tone control filter (tilt EQ)
        tone = self.parameters['tone']
        mid_freq = 1000 / nyquist
        mid_freq = np.clip(mid_freq, 0.001, 0.99)
        
        if tone > 0.5:
            # Boost highs
            boost_db = (tone - 0.5) * 12  # Up to 6dB boost
            self.tone_b, self.tone_a = signal.iirpeak(mid_freq, Q=0.7, gain=boost_db)
        else:
            # Cut highs
            cut_db = (0.5 - tone) * -12  # Up to -6dB cut
            self.tone_b, self.tone_a = signal.iirnotch(mid_freq, Q=0.7, gain=cut_db)
    
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio through cyberpunk distortion"""
        if len(audio_data) == 0:
            return audio_data
        
        # Create a copy to avoid modifying input
        processed = audio_data.copy().astype(np.float32)
        
        # Input gain staging
        drive = self.parameters['drive']
        processed *= (1.0 + drive * 3.0)  # Up to 4x gain
        
        # Pre-distortion filtering (high-pass)
        processed = self._apply_filter(processed, self.hp_b, self.hp_a, 'hp')
        
        # Digital bit crushing
        bit_crush = self.parameters['bit_crush']
        if bit_crush > 0.01:
            processed = self._apply_bit_crush(processed, bit_crush)
        
        # Non-linear distortion
        processed = self._apply_distortion(processed, drive)
        
        # Analog-style saturation
        saturation = self.parameters['saturation']
        if saturation > 0.01:
            processed = self._apply_saturation(processed, saturation)
        
        # Tone shaping
        processed = self._apply_filter(processed, self.tone_b, self.tone_a, 'tone')
        
        # Post-distortion filtering (low-pass)
        processed = self._apply_filter(processed, self.lp_b, self.lp_a, 'lp')
        
        # Output gain and limiting
        output_gain = self.parameters['output_gain']
        processed *= output_gain
        
        # Soft limiting to prevent clipping
        processed = np.tanh(processed * 0.8) * 0.9
        
        return processed
    
    def _apply_filter(self, audio_data: np.ndarray, b_coeffs: np.ndarray, a_coeffs: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply IIR filter with state preservation"""
        try:
            # Use lfilter for real-time processing
            return signal.lfilter(b_coeffs, a_coeffs, audio_data)
        except Exception:
            # Fallback if filtering fails
            return audio_data
    
    def _apply_bit_crush(self, audio_data: np.ndarray, intensity: float) -> np.ndarray:
        """Apply digital bit crushing effect"""
        # Reduce bit depth
        bit_depth = 16 - int(intensity * 12)  # From 16-bit down to 4-bit
        bit_depth = max(bit_depth, 4)
        
        # Quantize signal
        max_val = 2 ** (bit_depth - 1)
        quantized = np.round(audio_data * max_val) / max_val
        
        # Add some digital noise
        noise_level = intensity * 0.02
        noise = np.random.uniform(-noise_level, noise_level, len(audio_data))
        
        return quantized + noise
    
    def _apply_distortion(self, audio_data: np.ndarray, drive: float) -> np.ndarray:
        """Apply non-linear distortion"""
        # Asymmetric clipping for more aggressive sound
        threshold = 1.0 - drive * 0.7
        
        # Positive clipping (harder)
        pos_mask = audio_data > threshold
        audio_data[pos_mask] = threshold + np.tanh((audio_data[pos_mask] - threshold) * 5) * 0.2
        
        # Negative clipping (softer)
        neg_threshold = -threshold * 0.8
        neg_mask = audio_data < neg_threshold
        audio_data[neg_mask] = neg_threshold + np.tanh((audio_data[neg_mask] - neg_threshold) * 3) * 0.3
        
        # Harmonic generation
        if drive > 0.3:
            harmonic_gain = (drive - 0.3) * 0.2
            harmonics = np.sin(audio_data * np.pi * 2) * harmonic_gain
            audio_data += harmonics
        
        return audio_data
    
    def _apply_saturation(self, audio_data: np.ndarray, intensity: float) -> np.ndarray:
        """Apply analog-style saturation with memory"""
        # Tape-like saturation with hysteresis
        saturation_gain = 1.0 + intensity * 2.0
        saturated = np.tanh(audio_data * saturation_gain) * 0.8
        
        # Add some warmth with even harmonics
        warmth = np.sin(audio_data * np.pi) * intensity * 0.1
        
        return saturated + warmth
    
    def on_parameter_changed(self, name: str, old_value: Any, new_value: Any):
        """Update internal state when parameters change"""
        if name in ['tone', 'low_cut', 'high_cut']:
            self._update_filters()
    
    def on_enable(self):
        """Called when effect is enabled"""
        self._initialize_state()
    
    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get a named preset"""
        presets = {
            'light': {
                'drive': 0.3,
                'tone': 0.7,
                'output_gain': 0.9,
                'bit_crush': 0.1,
                'saturation': 0.2,
                'low_cut': 100,
                'high_cut': 6000,
            },
            'medium': {
                'drive': 0.6,
                'tone': 0.6,
                'output_gain': 0.8,
                'bit_crush': 0.3,
                'saturation': 0.4,
                'low_cut': 80,
                'high_cut': 7000,
            },
            'heavy': {
                'drive': 0.8,
                'tone': 0.5,
                'output_gain': 0.7,
                'bit_crush': 0.5,
                'saturation': 0.6,
                'low_cut': 60,
                'high_cut': 8000,
            },
            'cyberpunk': {
                'drive': 0.9,
                'tone': 0.4,
                'output_gain': 0.6,
                'bit_crush': 0.7,
                'saturation': 0.3,
                'low_cut': 120,
                'high_cut': 5000,
            }
        }
        
        return presets.get(preset_name, presets['medium'])
    
    def load_preset(self, preset_name: str):
        """Load a named preset"""
        preset = self.get_preset(preset_name)
        for param_name, value in preset.items():
            if param_name in self.parameters:
                self.set_parameter(param_name, value)
