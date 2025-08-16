"""
Basic Distortion Effect - Simple, warm distortion for general guitar playing
"""

import numpy as np
from scipy import signal
from typing import Dict, Any

from effects.base_effect import BaseEffect

class BasicDistortionEffect(BaseEffect):
    """Basic distortion effect for general guitar playing"""
    
    def _initialize_parameters(self, **kwargs):
        """Initialize basic distortion parameters"""
        self.parameters = {
            'gain': kwargs.get('gain', 0.4),              # Input gain (0.0 - 1.0)
            'drive': kwargs.get('drive', 0.3),            # Distortion amount (0.0 - 1.0)
            'tone': kwargs.get('tone', 0.6),              # Tone control (0.0 - 1.0)
            'output_level': kwargs.get('output_level', 0.7),  # Output level (0.0 - 1.0)
            'warmth': kwargs.get('warmth', 0.2),          # Analog warmth (0.0 - 1.0)
        }
    
    def _initialize_state(self):
        """Initialize distortion state"""
        self.state = {
            'filter_state': np.zeros(4),     # For tone filtering
        }
        
        # Create tone filter coefficients
        self._update_tone_filter()
    
    def _update_tone_filter(self):
        """Update tone filter coefficients"""
        nyquist = self.sample_rate / 2
        tone = self.parameters['tone']
        
        # Simple tone control - high shelf filter
        freq = 2000 / nyquist  # 2kHz shelf
        freq = np.clip(freq, 0.001, 0.99)
        
        # Create high shelf filter
        gain_db = (tone - 0.5) * 12  # ±6dB adjustment
        if abs(gain_db) > 0.1:
            # Use a simple high-frequency emphasis/de-emphasis
            if gain_db > 0:
                # Boost highs
                self.tone_b, self.tone_a = signal.butter(1, freq, 'highpass')
                self.tone_gain = 1.0 + (gain_db / 12.0)
            else:
                # Cut highs  
                self.tone_b, self.tone_a = signal.butter(1, freq, 'lowpass')
                self.tone_gain = 1.0
        else:
            # No filtering
            self.tone_b, self.tone_a = [1.0], [1.0]
            self.tone_gain = 1.0
    
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio through basic distortion"""
        if len(audio_data) == 0:
            return audio_data
        
        # Create a copy to avoid modifying input
        processed = audio_data.copy().astype(np.float32)
        
        # Input gain
        gain = self.parameters['gain']
        processed *= (1.0 + gain * 2.0)  # Up to 3x gain
        
        # Soft clipping distortion
        drive = self.parameters['drive']
        if drive > 0.01:
            processed = self._apply_soft_distortion(processed, drive)
        
        # Add harmonic warmth
        warmth = self.parameters['warmth']
        if warmth > 0.01:
            processed = self._add_warmth(processed, warmth)
        
        # Tone shaping
        processed = self._apply_tone_filter(processed)
        
        # Output level
        output_level = self.parameters['output_level']
        processed *= output_level
        
        # Final soft limiting
        processed = np.tanh(processed * 0.9) * 0.8
        
        return processed
    
    def _apply_soft_distortion(self, audio_data: np.ndarray, drive: float) -> np.ndarray:
        """Apply gentle soft clipping distortion"""
        # Gentle symmetric soft clipping
        threshold = 1.0 - drive * 0.5
        
        # Soft clipping using tanh
        clipping_factor = 1.0 + drive * 3.0
        distorted = np.tanh(audio_data * clipping_factor) * threshold
        
        # Blend with clean signal for subtlety
        blend = 0.6 + drive * 0.4  # 60% to 100% distorted signal
        result = audio_data * (1.0 - blend) + distorted * blend
        
        return result
    
    def _add_warmth(self, audio_data: np.ndarray, warmth: float) -> np.ndarray:
        """Add analog-style warmth with gentle even harmonics"""
        # Generate subtle even harmonics
        even_harmonics = np.sin(audio_data * np.pi) * warmth * 0.05
        
        # Add subtle saturation
        saturated = np.tanh(audio_data * (1.0 + warmth * 0.5)) * 0.95
        
        # Combine original with warmth
        return audio_data * (1.0 - warmth * 0.3) + saturated * warmth * 0.3 + even_harmonics
    
    def _apply_tone_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply tone filtering"""
        try:
            if hasattr(self, 'tone_b') and hasattr(self, 'tone_a'):
                filtered = signal.lfilter(self.tone_b, self.tone_a, audio_data)
                return filtered * self.tone_gain
            else:
                return audio_data
        except Exception:
            return audio_data
    
    def on_parameter_changed(self, name: str, old_value: Any, new_value: Any):
        """Update internal state when parameters change"""
        if name == 'tone':
            self._update_tone_filter()
    
    def on_enable(self):
        """Called when effect is enabled"""
        self._initialize_state()
    
    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get a named preset"""
        presets = {
            'clean': {
                'gain': 0.1,
                'drive': 0.1,
                'tone': 0.6,
                'output_level': 0.8,
                'warmth': 0.1,
            },
            'light': {
                'gain': 0.3,
                'drive': 0.2,
                'tone': 0.7,
                'output_level': 0.7,
                'warmth': 0.2,
            },
            'medium': {
                'gain': 0.5,
                'drive': 0.4,
                'tone': 0.6,
                'output_level': 0.6,
                'warmth': 0.3,
            },
            'warm': {
                'gain': 0.4,
                'drive': 0.3,
                'tone': 0.5,
                'output_level': 0.7,
                'warmth': 0.5,
            }
        }
        
        return presets.get(preset_name, presets['light'])
    
    def load_preset(self, preset_name: str):
        """Load a named preset"""
        preset = self.get_preset(preset_name)
        for param_name, value in preset.items():
            if param_name in self.parameters:
                self.set_parameter(param_name, value)