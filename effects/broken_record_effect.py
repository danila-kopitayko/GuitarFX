"""
Broken Record Effect - Vinyl-style glitch effect for pinch harmonics
"""

import numpy as np
from scipy import signal
from typing import Dict, Any

from effects.base_effect import BaseEffect

class BrokenRecordEffect(BaseEffect):
    """Broken record/vinyl glitch effect for pinch harmonics"""
    
    def _initialize_parameters(self, **kwargs):
        """Initialize broken record parameters"""
        self.parameters = {
            'scratch_intensity': kwargs.get('scratch_intensity', 0.7),    # Scratch effect intensity (0.0 - 1.0)
            'wow_flutter': kwargs.get('wow_flutter', 0.5),               # Pitch variation (0.0 - 1.0)
            'crackle': kwargs.get('crackle', 0.4),                       # Surface noise (0.0 - 1.0)
            'skip_probability': kwargs.get('skip_probability', 0.1),      # Chance of skipping (0.0 - 1.0)
            'reverse_probability': kwargs.get('reverse_probability', 0.05), # Chance of reverse (0.0 - 1.0)
            'pitch_bend': kwargs.get('pitch_bend', 0.6),                 # Pitch bending amount (0.0 - 1.0)
            'output_gain': kwargs.get('output_gain', 0.8),               # Output level (0.0 - 1.0)
            'vintage_filter': kwargs.get('vintage_filter', 0.5),         # Vintage filtering (0.0 - 1.0)
        }
    
    def _initialize_state(self):
        """Initialize broken record state"""
        self.state = {
            'buffer': np.zeros(8192),          # Circular buffer for effects
            'buffer_index': 0,                 # Current buffer position
            'playback_rate': 1.0,             # Current playback rate
            'pitch_lfo_phase': 0.0,           # LFO for pitch modulation
            'scratch_buffer': np.zeros(2048), # Buffer for scratch effects
            'scratch_index': 0,               # Scratch buffer position
            'skip_countdown': 0,              # Frames until next skip
            'reverse_countdown': 0,           # Frames in reverse mode
            'crackle_state': np.random.RandomState(42),  # Random state for crackle
            'filter_history': np.zeros(4),    # Filter memory
        }
        
        # Generate scratch samples
        self._generate_scratch_samples()
        
        # Create vintage filter
        self._update_vintage_filter()
    
    def _generate_scratch_samples(self):
        """Generate scratch sound samples"""
        # Create vinyl scratch-like noise
        scratch_length = len(self.state['scratch_buffer'])
        
        # Brown noise for base scratch sound
        brown_noise = np.cumsum(np.random.randn(scratch_length)) * 0.1
        brown_noise = brown_noise / np.max(np.abs(brown_noise))
        
        # Add high-frequency scratching
        high_freq_noise = np.random.randn(scratch_length) * 0.3
        
        # Filter for characteristic scratch sound
        nyquist = self.sample_rate / 2
        scratch_freq = 2000 / nyquist
        b, a = signal.butter(2, scratch_freq, 'highpass')
        filtered_scratch = signal.lfilter(b, a, brown_noise + high_freq_noise)
        
        # Apply envelope for more realistic scratch
        envelope = np.abs(np.sin(np.linspace(0, np.pi * 4, scratch_length))) ** 0.5
        
        self.state['scratch_buffer'] = filtered_scratch * envelope * 0.5
    
    def _update_vintage_filter(self):
        """Update vintage-style filter coefficients"""
        # Create vintage vinyl filter response
        vintage_amount = self.parameters['vintage_filter']
        
        if vintage_amount < 0.01:
            self.vintage_b = np.array([1.0])
            self.vintage_a = np.array([1.0])
            return
        
        # Roll off highs like old vinyl
        nyquist = self.sample_rate / 2
        cutoff_freq = 8000 - (vintage_amount * 4000)  # 8kHz down to 4kHz
        cutoff_norm = cutoff_freq / nyquist
        cutoff_norm = np.clip(cutoff_norm, 0.1, 0.95)
        
        # Low-pass filter with resonance
        self.vintage_b, self.vintage_a = signal.butter(2, cutoff_norm, 'lowpass')
        
        # Add some mid boost for vintage character
        mid_freq = 800 / nyquist
        mid_boost = vintage_amount * 3  # Up to 3dB boost
        mid_b, mid_a = signal.iirpeak(mid_freq, Q=1.5, gain=mid_boost)
        
        # Combine filters
        combined_b = np.convolve(self.vintage_b, mid_b)
        combined_a = np.convolve(self.vintage_a, mid_a)
        self.vintage_b = combined_b
        self.vintage_a = combined_a
    
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio through broken record effect"""
        if len(audio_data) == 0:
            return audio_data
        
        # Create a copy to avoid modifying input
        processed = audio_data.copy().astype(np.float32)
        
        # Apply wow and flutter (pitch modulation)
        processed = self._apply_wow_flutter(processed)
        
        # Apply random skips and stutters
        processed = self._apply_skips_and_stutters(processed)
        
        # Add scratch sounds
        processed = self._apply_scratches(processed)
        
        # Add surface crackle
        processed = self._apply_crackle(processed)
        
        # Apply vintage filtering
        processed = self._apply_vintage_filter(processed)
        
        # Apply output gain
        output_gain = self.parameters['output_gain']
        processed *= output_gain
        
        # Soft limiting
        processed = np.tanh(processed * 0.9)
        
        return processed
    
    def _apply_wow_flutter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply wow and flutter pitch modulation"""
        wow_flutter = self.parameters['wow_flutter']
        if wow_flutter < 0.01:
            return audio_data
        
        # Create LFO for pitch modulation
        lfo_freq = 0.5 + wow_flutter * 2  # 0.5 to 2.5 Hz
        lfo_samples = len(audio_data)
        
        phase_increment = 2 * np.pi * lfo_freq / self.sample_rate
        phases = self.state['pitch_lfo_phase'] + np.arange(lfo_samples) * phase_increment
        
        # Multiple LFOs for complex modulation
        wow_lfo = np.sin(phases) * 0.02 * wow_flutter  # Slow wow
        flutter_lfo = np.sin(phases * 8) * 0.005 * wow_flutter  # Fast flutter
        
        pitch_mod = 1.0 + wow_lfo + flutter_lfo
        
        # Update phase
        self.state['pitch_lfo_phase'] = phases[-1] % (2 * np.pi)
        
        # Apply pitch modulation using interpolation
        modulated = self._apply_pitch_modulation(audio_data, pitch_mod)
        
        return modulated
    
    def _apply_pitch_modulation(self, audio_data: np.ndarray, pitch_mod: np.ndarray) -> np.ndarray:
        """Apply pitch modulation using interpolation"""
        output = np.zeros_like(audio_data)
        buffer_size = len(self.state['buffer'])
        
        # Add new audio to circular buffer
        for i, sample in enumerate(audio_data):
            self.state['buffer'][self.state['buffer_index']] = sample
            self.state['buffer_index'] = (self.state['buffer_index'] + 1) % buffer_size
        
        # Read from buffer with pitch modulation
        for i in range(len(audio_data)):
            # Calculate read position with pitch modulation
            read_offset = pitch_mod[i] * 100  # Max 100 samples offset
            read_pos = (self.state['buffer_index'] - len(audio_data) + i - read_offset) % buffer_size
            
            # Linear interpolation
            pos_int = int(read_pos)
            pos_frac = read_pos - pos_int
            
            sample1 = self.state['buffer'][pos_int]
            sample2 = self.state['buffer'][(pos_int + 1) % buffer_size]
            
            output[i] = sample1 * (1 - pos_frac) + sample2 * pos_frac
        
        return output
    
    def _apply_skips_and_stutters(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply random skips and stutters"""
        skip_prob = self.parameters['skip_probability']
        reverse_prob = self.parameters['reverse_probability']
        
        if skip_prob < 0.01 and reverse_prob < 0.01:
            return audio_data
        
        output = audio_data.copy()
        
        for i in range(len(audio_data)):
            # Handle skip countdown
            if self.state['skip_countdown'] > 0:
                # During skip, repeat previous sample or add silence
                if i > 0:
                    output[i] = output[i-1] * 0.5  # Fade out
                else:
                    output[i] = 0
                self.state['skip_countdown'] -= 1
                continue
            
            # Handle reverse countdown
            if self.state['reverse_countdown'] > 0:
                # Reverse audio by reading from buffer backwards
                reverse_idx = len(audio_data) - 1 - (self.state['reverse_countdown'] % len(audio_data))
                if reverse_idx < len(audio_data):
                    output[i] = audio_data[reverse_idx] * 0.7
                self.state['reverse_countdown'] -= 1
                continue
            
            # Check for new skip
            if np.random.random() < skip_prob / self.sample_rate:
                skip_length = np.random.randint(50, 500)  # Skip 1-11ms
                self.state['skip_countdown'] = skip_length
                continue
            
            # Check for reverse
            if np.random.random() < reverse_prob / self.sample_rate:
                reverse_length = np.random.randint(100, 1000)  # Reverse 2-23ms
                self.state['reverse_countdown'] = reverse_length
                continue
        
        return output
    
    def _apply_scratches(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply vinyl scratch sounds"""
        scratch_intensity = self.parameters['scratch_intensity']
        if scratch_intensity < 0.01:
            return audio_data
        
        output = audio_data.copy()
        scratch_buffer = self.state['scratch_buffer']
        
        # Randomly trigger scratches
        for i in range(len(audio_data)):
            # Probability of scratch based on intensity
            if np.random.random() < scratch_intensity * 0.001:  # Rare but noticeable
                # Add scratch sound
                scratch_sample = scratch_buffer[self.state['scratch_index']]
                self.state['scratch_index'] = (self.state['scratch_index'] + 1) % len(scratch_buffer)
                
                # Mix scratch with audio
                output[i] = output[i] * 0.7 + scratch_sample * scratch_intensity
        
        return output
    
    def _apply_crackle(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply surface noise and crackle"""
        crackle = self.parameters['crackle']
        if crackle < 0.01:
            return audio_data
        
        # Generate crackle noise
        crackle_noise = self.state['crackle_state'].randn(len(audio_data)) * crackle * 0.05
        
        # Make it more vinyl-like with filtering
        if len(crackle_noise) > 1:
            # High-pass filter for crackle character
            crackle_noise = signal.lfilter([1, -0.95], [1], crackle_noise)
        
        # Add pops (rare, loud clicks)
        for i in range(len(audio_data)):
            if np.random.random() < crackle * 0.0001:  # Very rare
                pop_amplitude = np.random.uniform(0.1, 0.3) * crackle
                crackle_noise[i] += pop_amplitude
        
        return audio_data + crackle_noise
    
    def _apply_vintage_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply vintage-style filtering"""
        vintage_filter = self.parameters['vintage_filter']
        if vintage_filter < 0.01:
            return audio_data
        
        try:
            # Apply vintage filter
            filtered = signal.lfilter(self.vintage_b, self.vintage_a, audio_data)
            
            # Mix with original based on vintage amount
            return audio_data * (1 - vintage_filter) + filtered * vintage_filter
            
        except Exception:
            # Fallback if filtering fails
            return audio_data
    
    def on_parameter_changed(self, name: str, old_value: Any, new_value: Any):
        """Update internal state when parameters change"""
        if name == 'vintage_filter':
            self._update_vintage_filter()
        elif name in ['scratch_intensity', 'crackle']:
            # Regenerate scratch samples if needed
            if name == 'scratch_intensity':
                self._generate_scratch_samples()
    
    def on_enable(self):
        """Called when effect is enabled"""
        self._initialize_state()
        self._generate_scratch_samples()
    
    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get a named preset"""
        presets = {
            'light_wear': {
                'scratch_intensity': 0.2,
                'wow_flutter': 0.3,
                'crackle': 0.2,
                'skip_probability': 0.02,
                'reverse_probability': 0.01,
                'pitch_bend': 0.3,
                'output_gain': 0.9,
                'vintage_filter': 0.3,
            },
            'medium_wear': {
                'scratch_intensity': 0.5,
                'wow_flutter': 0.5,
                'crackle': 0.4,
                'skip_probability': 0.05,
                'reverse_probability': 0.02,
                'pitch_bend': 0.5,
                'output_gain': 0.8,
                'vintage_filter': 0.5,
            },
            'heavy_damage': {
                'scratch_intensity': 0.8,
                'wow_flutter': 0.7,
                'crackle': 0.6,
                'skip_probability': 0.1,
                'reverse_probability': 0.05,
                'pitch_bend': 0.7,
                'output_gain': 0.7,
                'vintage_filter': 0.7,
            },
            'broken': {
                'scratch_intensity': 1.0,
                'wow_flutter': 0.9,
                'crackle': 0.8,
                'skip_probability': 0.2,
                'reverse_probability': 0.1,
                'pitch_bend': 1.0,
                'output_gain': 0.6,
                'vintage_filter': 0.8,
            }
        }
        
        return presets.get(preset_name, presets['medium_wear'])
    
    def load_preset(self, preset_name: str):
        """Load a named preset"""
        preset = self.get_preset(preset_name)
        for param_name, value in preset.items():
            if param_name in self.parameters:
                self.set_parameter(param_name, value)
